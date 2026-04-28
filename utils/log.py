from datetime import datetime
import numpy as np

def log_full_matrix(f, name, mat):
    f.write(f"{name} shape: {mat.shape}\n")
    f.write(f"{name} values:\n")
    f.write(np.array2string(mat, precision=6, separator=', ', max_line_width=np.inf))
    f.write("\n\n")


def log_model(model):
    log_file = f"model_full_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    with open(log_file, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("FULL MLP WEIGHTS & BIASES DUMP\n")
        f.write("=" * 80 + "\n\n")

        for idx, layer in enumerate(model.layers):
            if hasattr(layer, "W"):
                f.write(f"[Layer {idx}] {layer.__class__.__name__}\n")
                f.write("-" * 80 + "\n")

                log_full_matrix(f, "Weights", layer.W)
                log_full_matrix(f, "Biases", layer.b)

                f.write("\n" + "=" * 80 + "\n\n")

    print(f"Full model saved to: {log_file}")

