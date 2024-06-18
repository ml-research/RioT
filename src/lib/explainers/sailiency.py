from typing import Callable
import torch

class HorizonSailiencyExplainer:

    def __init__(self,normalize: bool = True) -> None:
        self.normalize = normalize

    def attribute(self,input: torch.Tensor, predictions: torch.Tensor,horizon_tf: Callable[[torch.Tensor], torch.Tensor]=lambda x: x):
        A_gradX = torch.zeros_like(input)

        for i in range(predictions.shape[-1]):
            gradXes = torch.autograd.grad(predictions[...,i:i+1], input,torch.ones_like(predictions[...,i:i+1]), create_graph=True, allow_unused=True)[0]
            A_gradX_elem = horizon_tf(gradXes)
            A_gradX_elem = torch.clamp_min(A_gradX_elem, 0.0)
            A_gradX += A_gradX_elem
        if self.normalize:
            A_min = torch.min(A_gradX, dim=-1, keepdim=True).values
            A_max = torch.max(A_gradX, dim=-1, keepdim=True).values
            A_range = A_max - A_min + 1e-8

            A_gradX = (A_gradX - A_min) / A_range
        
        
                

        return A_gradX*1000