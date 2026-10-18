import torch, torchvision
dev = "cuda"
x = torch.rand(32, 3, 224, 224, device=dev)
u8 = (x * 255).round().clamp(0, 255).to(torch.uint8)
for q in [85, 90, 95, 99, 100]:
    try:
        torchvision.io.encode_jpeg(list(u8[:4]), quality=q); print("q", q, "OK")
    except Exception as e:
        print("q", q, "FAIL", e)
sub = u8[torch.tensor([1, 5, 7], device=dev)]
for name, imgs in [("index-select slice", [sub[j] for j in range(3)]), ("single", [u8[0]]), ("contig", [u8[j].contiguous() for j in range(3)])]:
    try:
        torchvision.io.encode_jpeg(imgs, quality=90); print(name, "OK")
    except Exception as e:
        print(name, "FAIL", e)
import torch.nn as nn
class M(nn.Module):
    def f(self):
        try:
            raise RuntimeError("x")
        except RuntimeError:
            self._flag = use = False
        return getattr(self, "_flag", True)
print("flag persists:", M().f())
