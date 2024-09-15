import torch
import torchvision
import torchaudio

# Test, ob die Pakete geladen wurden
print(f"PyTorch-Version: {torch.__version__}")
print(f"Torchvision-Version: {torchvision.__version__}")
print(f"Torchaudio-Version: {torchaudio.__version__}")

# Teste, ob CUDA verfügbar ist
print(f"CUDA verfügbar: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"cuDNN Version: {torch.backends.cudnn.version()}")

# Teste die GPU-Verfügbarkeit
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = torch.randn(3, 3).to(device)
print(f"Tensor läuft auf: {x.device}")
