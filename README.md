# ComfyUI-SeedreamOfficialInput

ComfyUI built-in Seedream node downscales reference images to 2048x2048 on upload.
This custom node calls the same ByteDance API, but uploads at official BytePlus limits:

- longest side <= 6000px
- total pixels <= 36MP
- file <= 30MB (JPEG if PNG would be huge)
- aspect 1:16 to 16:1

Output resolution is unchanged (Seedream 5.0 Pro still max ~2K / 4.19MP total pixels).

Search in ComfyUI: `Seedream 官方输入 (6000px)`

## Install

Clone into ComfyUI `custom_nodes`:

```text
ComfyUI/custom_nodes/ComfyUI-SeedreamOfficialInput
```

Restart ComfyUI / Comfy Desktop.

Uses the same Comfy Org / BytePlus API billing as the built-in Seedream node.

## License

MIT
