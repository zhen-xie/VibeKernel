import torch

class SeparatedSoftmaxBackend:
    name = "separated_softmax"
    def prepare(self, x):
        self.x = x; self.y = torch.empty_like(x); self.run(); torch.cuda.synchronize()
    def run(self):
        maximum = self.x.max(dim=1, keepdim=True).values
        exponentials = torch.exp(self.x - maximum)
        self.y.copy_(exponentials / exponentials.sum(dim=1, keepdim=True))
        return [self.y]
    def metadata(self): return {"implementation": "separate Max/Sub/Exp/Sum/Div operations", "kernel_count": 5}
    def close(self): pass
