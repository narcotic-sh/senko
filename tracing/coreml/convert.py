# Convert camplusplus_coreml.py into CoreML format
# Batch size 16, 150 frames of 80 dim fbank features

import torch
import coremltools as ct
import numpy as np
from pathlib import Path
from .camplusplus_coreml import CAMPPlusCoreML
from senko import CAMPPlus

def convert_campplus_to_coreml():
    print("=" * 60)
    print("CAM++ CoreML Converter")
    print("=" * 60)

    # Configuration
    BATCH_SIZE = 16
    FIXED_FRAMES = 150
    FEATURE_DIM = 80
    EMBEDDING_DIM = 192

    # Initialize PyTorch CoreML-friendly model
    coreml_friendly_model = CAMPPlusCoreML(feat_dim=FEATURE_DIM, embedding_size=EMBEDDING_DIM)

    # Load weights into CoreML-friendly model
    weights_path = "models/speech_campplus_sv_zh_en_16k-common_advanced/campplus_cn_en_common.pt"
    state_dict = torch.load(weights_path, map_location='cpu', weights_only=True)
    coreml_friendly_model.load_state_dict(state_dict)
    coreml_friendly_model.eval()
    print(f"✓ Loaded weights into CoreML-friendly model (CAMPPlusCoreML)")

    # Create example input
    example_input = torch.randn(BATCH_SIZE, FIXED_FRAMES, FEATURE_DIM)

    # Trace the CoreML-friendly model
    print("\nTracing CoreML-friendly model...")
    traced_model = torch.jit.trace(coreml_friendly_model, example_input)

    # Convert to CoreML
    print("\nConverting to CoreML...")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Fixed frames: {FIXED_FRAMES}")
    print(f"  Compute unit: CPU_AND_NE (Neural Engine)")

    input_type = ct.TensorType(
        name="input_features",
        shape=(BATCH_SIZE, FIXED_FRAMES, FEATURE_DIM),
        dtype=np.float32
    )

    coreml_model = ct.convert(
        traced_model,
        inputs=[input_type],
        outputs=[ct.TensorType(name="embeddings", dtype=np.float32)],
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS14,
    )

    # Add metadata
    coreml_model.author = "Original: 3D-Speaker / Speech Lab, Alibaba Group, CoreML conversion: Narcotic Software"
    coreml_model.short_description = "CAM++ speaker embedding model"
    coreml_model.input_description["input_features"] = f"Fbank features: ({BATCH_SIZE}, {FIXED_FRAMES}, {FEATURE_DIM})"
    coreml_model.output_description["embeddings"] = f"Speaker embeddings: ({BATCH_SIZE}, {EMBEDDING_DIM})"

    # Save the model
    output_path = "./tracing/camplusplus_batch16.mlpackage"
    coreml_model.save(output_path)
    print(f"\n✓ Saved CoreML model to: {output_path}")

    # Verify conversion accuracy against reference model
    print("\nVerifying conversion accuracy against reference model...")
    coreml_model_loaded = ct.models.MLModel(output_path)

    # Test with example input
    test_input = example_input.numpy()
    coreml_output = coreml_model_loaded.predict({'input_features': test_input})
    coreml_embeddings = coreml_output['embeddings']

    # Initialize reference model (original CAMPPlus)
    reference_model = CAMPPlus(feat_dim=FEATURE_DIM, embedding_size=EMBEDDING_DIM)

    # Load pretrained weights into reference model
    reference_model.load_state_dict(state_dict)
    reference_model.eval()
    print(f"✓ Loaded weights into reference model from: {weights_path}")

    # Test reference model
    print("\nTesting reference model...")
    with torch.no_grad():
        reference_output = reference_model(example_input)
    print(f"  Input shape: {example_input.shape}")
    print(f"  Reference output shape: {reference_output.shape}")

    # Compare CoreML output with reference model output
    reference_np = reference_output.numpy()
    max_diff = np.max(np.abs(reference_np - coreml_embeddings))
    mean_diff = np.mean(np.abs(reference_np - coreml_embeddings))

    # Calculate cosine similarity between CoreML and reference
    cosine_sims = []
    for i in range(BATCH_SIZE):
        ref_norm = reference_np[i] / (np.linalg.norm(reference_np[i]) + 1e-8)
        cm_norm = coreml_embeddings[i] / (np.linalg.norm(coreml_embeddings[i]) + 1e-8)
        cosine_sim = np.sum(ref_norm * cm_norm)
        cosine_sims.append(cosine_sim)

    avg_cosine = np.mean(cosine_sims)
    min_cosine = np.min(cosine_sims)

    print(f"  Max difference: {max_diff:.6f}")
    print(f"  Mean difference: {mean_diff:.6f}")
    print(f"  Avg cosine similarity: {avg_cosine:.6f}")
    print(f"  Min cosine similarity: {min_cosine:.6f}")

    if avg_cosine > 0.999:
        print("\n  ✓ Accuracy verification PASSED")
    elif avg_cosine > 0.99:
        print("\n  ⚠ Warning: Slightly lower accuracy than expected")
    else:
        print("\n  ⚠ Warning: Significant accuracy difference detected")

    print("\n" + "=" * 60)
    print("CONVERSION COMPLETE")
    print("=" * 60)
    print(f"Model: {output_path}")
    print(f"Batch size: {BATCH_SIZE}")

    return output_path


if __name__ == "__main__":
    convert_campplus_to_coreml()