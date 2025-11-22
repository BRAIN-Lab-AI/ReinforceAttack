import argparse, torch
from PIL import Image
from torchvision import transforms, models
from PatchAttack.PatchAttack_config import configure_PA
from PatchAttack.PatchAttack_attackers import TPA

def load_img(path):
    tfm = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    return tfm(Image.open(path).convert("RGB"))

def main():
    ap = argparse.ArgumentParser(description="Shape-aware PatchAttack (equal-area shapes).")
    ap.add_argument("--image", required=True, default= 'Images\n01632777_10139.JPEG' , help="path to RGB image")
    ap.add_argument("--target", type=int, default=723, help="target class index")
    ap.add_argument("--area", type=float, default=0.02, help="per-occluder area fraction")
    ap.add_argument("--agents", type=int, default=10, help="# of placements (agents)")
    ap.add_argument("--steps", type=int, default=20, help="RL steps")
    ap.add_argument("--rl_batch", type=int, default=200, help="RL batch size")
    ap.add_argument("--lambda_area", type=float, default=0.0, help="area penalty λ")
    ap.add_argument("--run_title", default="ShapeAware_TPA")
    ap.add_argument("--tname", default="ImageNet_ILSVRC2012")
    args = ap.parse_args()

    # Configure PatchAttack for equal-area & shape-aware search
    configure_PA(
        t_name=args.tname, t_labels=[args.target], target=True,
        area_occlu=args.area, n_occlu=1,
        rl_batch=args.rl_batch, steps=args.steps, TPA_n_agents=args.agents,
        shape=None,                 # mixed shapes (shape-aware)
        lambda_area=args.lambda_area
    )

    # Victim model + input
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT).eval()
    x = load_img(args.image)
    with torch.no_grad():
        pred = model(x.unsqueeze(0)).softmax(1).argmax().item()
    y = torch.tensor([pred]).long()

    # Attack — force a fresh run every time
    attacker = TPA(dir_title=args.run_title)
    _imgs, record = attacker.attack(
        model=model, input_tensor=x, label_tensor=y,
        target=args.target, input_name="example",
        run_tag="new"  # always start fresh
    )

if __name__ == "__main__":
    main()
