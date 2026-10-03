"""Seedream node that uploads reference images at BytePlus official size limits.

ComfyUI's built-in ByteDance Seedream node silently downscales uploads to 2048x2048.
This node keeps the same API / billing, but prepares inputs to:
  - longest side <= 6000px
  - total pixels <= 36MP (6000x6000)
  - file <= 30MB (JPEG fallback for large PNGs)
  - aspect ratio 1:16 .. 16:1
"""

from typing_extensions import override

import numpy as np
import scipy.ndimage
import torch
from comfy_api.latest import IO, ComfyExtension, Input
import node_helpers
from comfy_extras.nodes_mask import composite
from comfy_api_nodes.apis.bytedance import (
    ImageTaskCreationResponse,
    Seedream4Options,
    Seedream4TaskCreationRequest,
    Seedream5OptimizePromptOptions,
)
from comfy_api_nodes.nodes_bytedance import (
    BYTEPLUS_IMAGE_ENDPOINT,
    SEEDREAM_MODELS,
    SEEDREAM_PRESETS,
    ByteDanceSeedreamNodeV2,
    get_image_url_from_response,
)
from comfy_api_nodes.util import (
    ApiEndpoint,
    download_url_to_image_tensor,
    downscale_image_tensor,
    downscale_image_tensor_by_max_side,
    get_number_of_images,
    sync_op,
    upload_images_to_comfyapi,
    validate_image_aspect_ratio,
    validate_string,
)

SEEDREAM_MAX_INPUT_SIDE = 6000
SEEDREAM_MAX_INPUT_PIXELS = 6000 * 6000
SEEDREAM_PRO_MAX_OUTPUT_SIDE = 4096


def _prepare_seedream_reference_image(image: torch.Tensor) -> torch.Tensor:
    if image.ndim == 4 and image.shape[0] > 1:
        return torch.stack([_prepare_seedream_reference_image(image[i]) for i in range(image.shape[0])])
    squeeze = image.ndim == 3
    if squeeze:
        image = image.unsqueeze(0)
    image = downscale_image_tensor_by_max_side(image, max_side=SEEDREAM_MAX_INPUT_SIDE)
    image = downscale_image_tensor(image, total_pixels=SEEDREAM_MAX_INPUT_PIXELS)
    return image.squeeze(0) if squeeze else image


def _seedream_reference_pixels(image: torch.Tensor) -> int:
    if image.ndim == 4:
        return int(image.shape[1] * image.shape[2])
    return int(image.shape[0] * image.shape[1])


async def _upload_seedream_reference_images(
    cls: type[IO.ComfyNode],
    images: torch.Tensor | list[torch.Tensor],
    *,
    max_images: int,
) -> list[str]:
    prepared: list[torch.Tensor] = []
    if isinstance(images, list):
        prepared.extend(_prepare_seedream_reference_image(img) for img in images)
    else:
        prepared.append(_prepare_seedream_reference_image(images))
    for tensor in prepared:
        validate_image_aspect_ratio(tensor, (1, 16), (16, 1), strict=False)
    mime = "image/png"
    if any(_seedream_reference_pixels(t) > 8_000_000 for t in prepared):
        mime = "image/jpeg"
    return await upload_images_to_comfyapi(
        cls,
        prepared,
        max_images=max_images,
        mime_type=mime,
        total_pixels=None,
        wait_label="Uploading reference images (official 6000px cap)",
    )


class SeedreamOfficialInput(ByteDanceSeedreamNodeV2):
    @classmethod
    def define_schema(cls):
        schema = ByteDanceSeedreamNodeV2.define_schema()
        schema.node_id = "SeedreamOfficialInput"
        schema.display_name = "Seedream 官方输入 (6000px)"
        schema.category = "image/ByteDance"
        schema.search_aliases = [
            "seedream",
            "seedream 5.0",
            "official input",
            "6000",
            "36mp",
            "高清输入",
            "官方输入",
        ]
        schema.description = (
            "Same ByteDance Seedream 4.5 / 5.0 API as the built-in node, but reference images "
            "are uploaded at official BytePlus limits (max side 6000px, 36MP, 30MB) instead of "
            "ComfyUI's silent 2048x2048 downscale. Output size is unchanged (Pro still max ~2K)."
        )
        # Built-in Pro widgets cap height at 2496, which clips 9:16 at 4.19MP (1536x2730).
        model_input = next((inp for inp in schema.inputs if inp.id == "model"), None)
        if model_input is not None:
            for opt in model_input.options or []:
                if getattr(opt, "key", None) == "seedream 5.0 pro":
                    for nested in opt.inputs or []:
                        if nested.id in ("width", "height"):
                            nested.max = SEEDREAM_PRO_MAX_OUTPUT_SIDE
        return schema

    @classmethod
    async def execute(
        cls,
        prompt: str,
        model: dict,
        seed: int = 0,
        watermark: bool = False,
        thinking: bool = True,
    ) -> IO.NodeOutput:
        validate_string(prompt, strip_whitespace=True, min_length=1)
        model_id = SEEDREAM_MODELS[model["model"]]
        presets = SEEDREAM_PRESETS[model_id]
        is_pro = "seedream-5-0-pro" in model_id

        size_preset = model.get("size_preset", presets[0][0])
        width = model.get("width", 2048)
        height = model.get("height", 2048)
        max_images = model.get("max_images", 1)
        sequential_image_generation = "disabled" if max_images == 1 else "auto"
        images_dict = model.get("images") or {}
        fail_on_partial = model.get("fail_on_partial", False)

        w = h = None
        for label, tw, th in presets:
            if label == size_preset:
                w, h = tw, th
                break
        if w is None or h is None:
            w, h = width, height

        out_num_pixels = w * h
        mp_provided = out_num_pixels / 1_000_000.0
        if is_pro:
            if out_num_pixels < 921_600:
                raise ValueError(
                    f"Minimum image resolution for the selected model is 0.92MP, but {mp_provided:.2f}MP provided."
                )
            if out_num_pixels > 4_194_304:
                raise ValueError(
                    f"Maximum image resolution for the selected model is 4.19MP, but {mp_provided:.2f}MP provided."
                )
        else:
            if ("seedream-4-5" in model_id or "seedream-5-0" in model_id) and out_num_pixels < 3_686_400:
                raise ValueError(
                    f"Minimum image resolution for the selected model is 3.68MP, but {mp_provided:.2f}MP provided."
                )
            if "seedream-4-0" in model_id and out_num_pixels < 921_600:
                raise ValueError(
                    f"Minimum image resolution that the selected model can generate is 0.92MP, "
                    f"but {mp_provided:.2f}MP provided."
                )
            if out_num_pixels > 16_777_216:
                raise ValueError(
                    f"Maximum image resolution for the selected model is 16.78MP, but {mp_provided:.2f}MP provided."
                )

        image_tensors: list[Input.Image] = [t for t in images_dict.values() if t is not None]
        n_input_images = sum(get_number_of_images(t) for t in image_tensors)
        max_num_of_images = 14 if model_id == "seedream-5-0-260128" else 10
        if n_input_images > max_num_of_images:
            raise ValueError(
                f"Maximum of {max_num_of_images} reference images are supported, but {n_input_images} received."
            )
        if sequential_image_generation == "auto" and n_input_images + max_images > 15:
            raise ValueError(
                "The maximum number of generated images plus the number of reference images cannot exceed 15."
            )
        if not thinking and n_input_images > 0:
            raise ValueError(
                "'thinking' can only be disabled for text-to-image; enable it when using reference images."
            )

        reference_images_urls: list[str] = []
        if image_tensors:
            reference_images_urls = await _upload_seedream_reference_images(
                cls,
                image_tensors,
                max_images=n_input_images,
            )

        optimize_prompt_options = None
        if n_input_images == 0:
            optimize_prompt_options = Seedream5OptimizePromptOptions(thinking="enabled" if thinking else "disabled")
        response = await sync_op(
            cls,
            ApiEndpoint(path=BYTEPLUS_IMAGE_ENDPOINT, method="POST"),
            response_model=ImageTaskCreationResponse,
            data=Seedream4TaskCreationRequest(
                model=model_id,
                prompt=prompt,
                image=reference_images_urls,
                size=f"{w}x{h}",
                seed=seed,
                sequential_image_generation=None if is_pro else sequential_image_generation,
                sequential_image_generation_options=None if is_pro else Seedream4Options(max_images=max_images),
                watermark=watermark,
                optimize_prompt_options=optimize_prompt_options,
            ),
        )
        if len(response.data) == 1:
            return IO.NodeOutput(await download_url_to_image_tensor(get_image_url_from_response(response)))
        urls = [str(d["url"]) for d in response.data if isinstance(d, dict) and "url" in d]
        if fail_on_partial and len(urls) < len(response.data):
            raise RuntimeError(f"Only {len(urls)} of {len(response.data)} images were generated before error.")
        return IO.NodeOutput(torch.cat([await download_url_to_image_tensor(i) for i in urls]))


def _mask_to_hw(mask: torch.Tensor, height: int, width: int) -> torch.Tensor:
    mask = mask.reshape((-1, mask.shape[-2], mask.shape[-1])).float()
    if mask.shape[-2] != height or mask.shape[-1] != width:
        mask = torch.nn.functional.interpolate(
            mask.unsqueeze(1), size=(height, width), mode="bilinear"
        ).squeeze(1)
    return mask.clamp(0.0, 1.0)


def _grow_blur_mask(mask: torch.Tensor, grow: int, blur: int) -> torch.Tensor:
    kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
    out = []
    for plane in mask.reshape((-1, mask.shape[-2], mask.shape[-1])):
        arr = plane.detach().float().cpu().numpy()
        for _ in range(abs(int(grow))):
            if grow > 0:
                arr = scipy.ndimage.grey_dilation(arr, footprint=kernel)
            else:
                arr = scipy.ndimage.grey_erosion(arr, footprint=kernel)
        if blur > 0:
            arr = scipy.ndimage.gaussian_filter(arr, sigma=float(blur) / 2.0)
        out.append(torch.from_numpy(arr.astype(np.float32)))
    return torch.stack(out, dim=0).to(device=mask.device).clamp(0.0, 1.0)


class SeedreamEditComposite(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="SeedreamEditComposite",
            display_name="Seedream 遮罩贴回原图",
            category="image/ByteDance",
            search_aliases=[
                "seedream",
                "composite",
                "mask",
                "inpaint",
                "贴回",
                "合成",
                "遮罩",
            ],
            description=(
                "Seedream has no mask input and redraws the whole image. This pastes the "
                "generated result back onto the original using the mask (white = keep generated). "
                "Empty / missing mask passes the generated image through unchanged."
            ),
            inputs=[
                IO.Image.Input("original", tooltip="Unedited source image."),
                IO.Image.Input("generated", tooltip="Seedream output."),
                IO.Mask.Input(
                    "mask",
                    optional=True,
                    tooltip="White = paste generated pixels. Empty mask = pass generated through.",
                ),
                IO.Mask.Input(
                    "mask_2",
                    optional=True,
                    tooltip="Optional second mask, OR-merged with mask (e.g. LoadImage + Painter).",
                ),
                IO.Int.Input(
                    "grow",
                    default=8,
                    min=-256,
                    max=256,
                    tooltip="Expand (positive) or shrink the mask before blending.",
                ),
                IO.Int.Input(
                    "blur",
                    default=12,
                    min=0,
                    max=256,
                    tooltip="Gaussian blur on the mask edge, in pixels.",
                ),
            ],
            outputs=[IO.Image.Output()],
        )

    @classmethod
    def execute(
        cls,
        original: torch.Tensor,
        generated: torch.Tensor,
        mask=None,
        mask_2=None,
        grow: int = 8,
        blur: int = 12,
    ) -> IO.NodeOutput:
        if original.ndim == 3:
            original = original.unsqueeze(0)
        if generated.ndim == 3:
            generated = generated.unsqueeze(0)
        dest_h, dest_w = int(original.shape[1]), int(original.shape[2])

        combined = None
        for candidate in (mask, mask_2):
            if candidate is None:
                continue
            prepared = _mask_to_hw(candidate, dest_h, dest_w)
            combined = prepared if combined is None else torch.maximum(combined, prepared)

        if combined is None or float(combined.max()) < 1e-3:
            return IO.NodeOutput(generated)

        if grow != 0 or blur > 0:
            combined = _grow_blur_mask(combined, grow, blur)

        original, generated = node_helpers.image_alpha_fix(original, generated)
        destination = original.clone().movedim(-1, 1)
        source = generated.movedim(-1, 1)
        output = composite(destination, source, 0, 0, combined, 1, True).movedim(1, -1)
        return IO.NodeOutput(output)


class SeedreamOfficialInputExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [SeedreamOfficialInput, SeedreamEditComposite]


async def comfy_entrypoint() -> SeedreamOfficialInputExtension:
    return SeedreamOfficialInputExtension()
