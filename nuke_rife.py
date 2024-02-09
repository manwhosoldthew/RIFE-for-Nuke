import logging
import torch

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


def trace_rife():
    from model.RIFE_HDv3 import Model

    model = Model()
    model.load_model("model", -1)
    LOGGER.info("Loaded v4.14 model.")
    model.eval()
    model.device()
    LOGGER.info(model)

    class FlowNetNuke(torch.nn.Module):
        def __init__(self, timestep: float = 0.5, scale: float = 1.0, optical_flow: int = 0):
            super().__init__()
            self.optical_flow = optical_flow
            self.timestep = timestep
            self.scale = scale
            self.flownet = model.flownet

        def forward(self, x):
            timestep = self.timestep
            scale = self.scale if self.scale in [0.125, 0.25, 0.5, 1.0, 2.0, 4.0] else 1.0
            b, c, h, w = x.shape
            device = torch.device("cuda") if x.is_cuda else torch.device("cpu")

            # Force input to float32
            if x.dtype != torch.float32:
                x = x.to(torch.float32)

            # Padding
            padding_factor = max(128, int(128 / scale))
            pad_h = ((h - 1) // padding_factor + 1) * padding_factor
            pad_w = ((w - 1) // padding_factor + 1) * padding_factor
            pad_dims = (0, pad_w - w, 0, pad_h - h)
            x = torch.nn.functional.pad(x, pad_dims)

            scale_list = (8.0 / scale, 4.0 / scale, 2.0 / scale, 1.0 / scale)
            flow, mask, image = self.flownet((x), timestep, scale_list)

            # Return the optical flow and mask
            if self.optical_flow:
                flow = flow[:, :, :h, :w]
                stmap_x = (
                    torch.linspace(0.0, 1.0, w, device=device).view(1, 1, 1, w).expand(1, -1, h, -1)
                )
                stmap_y = (
                    torch.linspace(1.0, 0.0, h, device=device).view(1, 1, h, 1).expand(1, -1, -1, w)
                )
                stmap_x_y = torch.cat([stmap_x, stmap_y, stmap_x, stmap_y], 1).to(device)

                displacement = torch.cat(
                    [
                        flow[:, 0:1, :, :] * 0.5 / ((w - 1.0) / 2.0),
                        flow[:, 1:2, :, :] * -0.5 / ((h - 1.0) / 2.0),
                        flow[:, 2:3, :, :] * 0.5 / ((w - 1.0) / 2.0),
                        flow[:, 3:4, :, :] * -0.5 / ((h - 1.0) / 2.0),
                    ],
                    1,
                )

                stmap = stmap_x_y + displacement

                stmap_packed = torch.zeros((b, 3, h, w), dtype=torch.float32)
                # Since Nuke 13 throws an error if we return an image with 5 channels,
                # we are packing both the flow channels into a single one and
                # returning a 3 channel image - stmap_fw, stmap_bw, and mask.

                # Scale numbers to use 11 and 12 bits respectively, shift and combine
                stmap_fw_x = (stmap[:, 0] * (2**11 - 1)).to(torch.int32)
                stmap_fw_y = (stmap[:, 1] * (2**12 - 1)).to(torch.int32)
                combined_fw = (stmap_fw_x << 12) | stmap_fw_y
                stmap_packed[:, 0] = torch.as_tensor(combined_fw, dtype=torch.float32)

                stmap_bw_x = (stmap[:, 2] * (2**11 - 1)).to(torch.int32)
                stmap_bw_y = (stmap[:, 3] * (2**12 - 1)).to(torch.int32)
                combined_bw = (stmap_bw_x << 12) | stmap_bw_y
                stmap_packed[:, 1] = torch.as_tensor(combined_bw, dtype=torch.float32)

                stmap_packed[:, 2] = mask[:, 0, :h, :w]
                return stmap_packed.contiguous()

            # Return the interpolated frames
            return image[:, :3, :h, :w].contiguous()

    model_file = "./nuke/Cattery/RIFE/RIFE_n13.pt"
    # model_file = "./nuke/Cattery/RIFE/RIFE_n13_ref.pt"
    with torch.jit.optimized_execution(True):
        rife_nuke = torch.jit.script(FlowNetNuke())
        rife_nuke.save(model_file)
        LOGGER.info(rife_nuke.code)
        LOGGER.info(rife_nuke.graph)
        LOGGER.info("Traced flow saved: %s", model_file)


if __name__ == "__main__":
    trace_rife()
