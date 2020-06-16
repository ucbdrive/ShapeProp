import torch
import torch.nn.functional as F
from torch.autograd import Function


class PeakStimulation(Function):

    @staticmethod
    def forward(ctx, input, win_size, peak_filter):
        ctx.num_flags = 3

        # peak finding
        assert win_size % 2 == 1, 'Window size for peak finding must be odd.'
        offset = (win_size - 1) // 2
        padding = torch.nn.ConstantPad2d(offset, float('-inf'))
        padded_maps = padding(input)
        batch_size, num_channels, h, w = padded_maps.size()
        element_map = torch.arange(0, h * w).long().view(1, 1, h, w)[:, :, offset: -offset, offset: -offset]
        element_map = element_map.to(input.device)
        _, indices  = F.max_pool2d(
            padded_maps,
            kernel_size = win_size, 
            stride = 1, 
            return_indices = True)
        peak_map = (indices == element_map)

        # peak filtering
        if peak_filter:
            mask = input >= peak_filter(input)
            peak_map = (peak_map & mask)

        # peak aggregation
        ctx.eps = 1e-5
        peak_map = peak_map.to(dtype=input.dtype)
        ctx.save_for_backward(input, peak_map)
        return (input * peak_map).view(batch_size, num_channels, -1).sum(2) / \
            (peak_map.view(batch_size, num_channels, -1).sum(2) + ctx.eps)

    @staticmethod
    def backward(ctx, grad_output):
        input, peak_map, = ctx.saved_tensors
        batch_size, num_channels, _, _ = input.size()
        grad_input = peak_map * grad_output.view(batch_size, num_channels, 1, 1) / \
            (peak_map.view(batch_size, num_channels, -1).sum(2).unsqueeze(2).unsqueeze(3) + ctx.eps)
        return (grad_input.to(dtype=input.dtype),) + (None,) * ctx.num_flags


def peak_stimulation(input, win_size=3, peak_filter=None):
    return PeakStimulation.apply(input, win_size, peak_filter)
