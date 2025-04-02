import os
import random
from PIL import Image
import itertools

# --- Robustness Tests ---

TEMPLATE_DIR = "templates"

def compose_template(food_path, bg_path=None, extra_path=None):
    food = Image.open(food_path).convert("RGBA")
    bg = Image.new("RGBA", food.size, (255, 255, 255, 255)) if bg_path is None else Image.open(bg_path).convert("RGBA")
    bg_w, bg_h = bg.size
    y_offset = int(bg_h * 0.05)

    food = food.resize((int(bg_w * 0.5), int(bg_h * 0.5)))
    fd_w, fd_h = food.size

    if extra_path:
        extra = Image.open(extra_path).convert("RGBA")
        extra = extra.resize((int(bg_w * 0.35), int(bg_h * 0.35)))
        ex_w, ex_h = extra.size
        bg.paste(extra, (bg_w - ex_w, bg_h - ex_h - y_offset), extra)

    bg.paste(food, ((bg_w - fd_w) // 2, bg_h - fd_h - y_offset), food)
    return bg.convert("RGB")

# Require 80% accuracy
def test_template_permutations(model, predict):
    food_classes = os.listdir(os.path.join(TEMPLATE_DIR, "food"))
    backgrounds = os.listdir(os.path.join(TEMPLATE_DIR, "background"))
    extras = os.listdir(os.path.join(TEMPLATE_DIR, "extras"))

    total_tests = 0
    passed_tests = 0
    failures = []

    for food_class in food_classes:
        class_index = int(food_class)
        food_dir = os.path.join(TEMPLATE_DIR, "food", food_class)
        food_items = os.listdir(food_dir)
        if not food_items:
            continue

        food_path = os.path.join(food_dir, random.choice(food_items))
        
        combinations = itertools.product(backgrounds, extras)
        for bg, extra in combinations:
                total_tests += 1
                img = compose_template(
                    food_path,
                    bg_path=os.path.join(TEMPLATE_DIR, "background", bg),
                    extra_path=os.path.join(TEMPLATE_DIR, "extras", extra)
                )
                pred = predict(model, img)
                if pred == class_index:
                    passed_tests += 1
                else:
                    failures.append((food_class, bg, extra, pred))

    pass_ratio = passed_tests / total_tests if total_tests > 0 else 0
    assert pass_ratio >= 0.80, f"Only {passed_tests}/{total_tests} ({pass_ratio*100:.1f}%) template permutations passed. Failures: {failures}"

