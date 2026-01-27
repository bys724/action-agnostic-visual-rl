#!/usr/bin/env python3
"""
모델 테스트 스크립트

Two-Stream Interleaved ViT 테스트

Usage:
    docker compose run --rm dev python src/test_models.py
"""

import torch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_two_stream_preprocessing():
    """Two-Stream Preprocessing 테스트"""
    print("\n" + "=" * 60)
    print("Testing Two-Stream Preprocessing")
    print("=" * 60)

    from src.models import TwoStreamPreprocessing

    preprocessing = TwoStreamPreprocessing(trainable_weights=True)

    img_prev = torch.rand(2, 3, 224, 224)
    img_curr = torch.rand(2, 3, 224, 224)

    # Combined output
    output = preprocessing(img_prev, img_curr)
    print(f"  Combined output: {output.shape}")  # [2, 6, 224, 224]
    assert output.shape == (2, 6, 224, 224)

    # Separate outputs
    m_out, p_out = preprocessing(img_prev, img_curr, return_separate=True)
    print(f"  M-channel: {m_out.shape}")  # [2, 4, 224, 224]
    print(f"  P-channel: {p_out.shape}")  # [2, 2, 224, 224]
    assert m_out.shape == (2, 4, 224, 224)
    assert p_out.shape == (2, 2, 224, 224)

    print(f"  Luminance weights: {preprocessing.get_learned_weights().tolist()}")

    print("\n✅ Two-Stream Preprocessing tests passed!")
    return True


def test_two_stream_interleaved_vit():
    """Two-Stream Interleaved ViT 테스트"""
    print("\n" + "=" * 60)
    print("Testing Two-Stream Interleaved ViT")
    print("=" * 60)

    from src.models import TwoStreamInterleavedViT, TwoStreamViTConfig

    # Config
    config = TwoStreamViTConfig(
        img_size=224,
        patch_size=14,
        embed_dim=512,
        num_heads=8,
        blocks_per_stage=4,
        num_stages=3,
        cross_attn_blocks=2,
        cross_attn_heads=4,
        output_dim=512,
    )

    print(f"  Config: embed_dim={config.embed_dim}, "
          f"blocks={config.blocks_per_stage}x{config.num_stages}, "
          f"cross_attn={config.cross_attn_blocks}")

    model = TwoStreamInterleavedViT(config)

    # Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,} ({total_params/1e6:.1f}M)")

    # Forward test
    img_prev = torch.rand(2, 3, 224, 224)
    img_curr = torch.rand(2, 3, 224, 224)

    output = model(img_prev, img_curr)
    print(f"  Input: prev={img_prev.shape}, curr={img_curr.shape}")
    print(f"  Output: {output.shape}")  # [2, 512]
    assert output.shape == (2, config.output_dim)

    # Separate CLS test
    cls_m, cls_p = model.get_separate_cls(img_prev, img_curr)
    print(f"  CLS_M: {cls_m.shape}, CLS_P: {cls_p.shape}")
    assert cls_m.shape == (2, config.embed_dim)
    assert cls_p.shape == (2, config.embed_dim)

    # Intermediates test
    output, intermediates = model(img_prev, img_curr, return_intermediates=True)
    print(f"  Intermediate stages: {len(intermediates)}")
    assert len(intermediates) == config.num_stages

    print("\n✅ Two-Stream Interleaved ViT tests passed!")
    return True


def test_gradient_flow():
    """Gradient flow 테스트"""
    print("\n" + "=" * 60)
    print("Testing Gradient Flow")
    print("=" * 60)

    from src.models import TwoStreamInterleavedViT, TwoStreamViTConfig

    # Smaller model for faster test
    config = TwoStreamViTConfig(
        embed_dim=256,
        num_heads=4,
        blocks_per_stage=2,
        num_stages=2,
        cross_attn_blocks=1,
    )
    model = TwoStreamInterleavedViT(config)

    img_prev = torch.rand(2, 3, 224, 224, requires_grad=True)
    img_curr = torch.rand(2, 3, 224, 224, requires_grad=True)

    output = model(img_prev, img_curr)
    loss = output.mean()
    loss.backward()

    print(f"  Input gradient exists: {img_prev.grad is not None}")
    print(f"  Input gradient mean: {img_prev.grad.mean().item():.6f}")

    has_grad = sum(1 for p in model.parameters() if p.grad is not None)
    total = sum(1 for p in model.parameters())
    print(f"  Parameters with gradients: {has_grad}/{total}")

    print("\n✅ Gradient flow test passed!")
    return True


def test_action_decoder():
    """Action Decoder 테스트"""
    print("\n" + "=" * 60)
    print("Testing Action Decoder")
    print("=" * 60)

    from src.models import ActionDecoder, ActionDecoderConfig

    config = ActionDecoderConfig(
        behavior_dim=512,
        hidden_dim=256,
        num_layers=3,
        action_dim=7,
        action_horizon=1,
        head_type="mlp",
    )

    decoder = ActionDecoder(config)

    behavior_embedding = torch.rand(2, 512)
    output = decoder(behavior_embedding)

    print(f"  Input: {behavior_embedding.shape}")
    print(f"  Output actions: {output['actions'].shape}")  # [2, 1, 7]
    assert output['actions'].shape == (2, 1, 7)

    # Loss test
    target = torch.rand(2, 1, 7)
    loss = decoder.compute_loss(behavior_embedding, target)
    print(f"  Loss: {loss.item():.4f}")

    print("\n✅ Action Decoder tests passed!")
    return True


def test_full_pipeline():
    """Full Pipeline 테스트"""
    print("\n" + "=" * 60)
    print("Testing Full Pipeline")
    print("=" * 60)

    from src.models import (
        TwoStreamInterleavedViT, TwoStreamViTConfig,
        ActionDecoder, ActionDecoderConfig, FullPipeline
    )

    # Encoder
    enc_config = TwoStreamViTConfig(
        embed_dim=256,
        num_heads=4,
        blocks_per_stage=2,
        num_stages=2,
        output_dim=256,
    )
    encoder = TwoStreamInterleavedViT(enc_config)

    # Decoder
    dec_config = ActionDecoderConfig(
        behavior_dim=256,
        hidden_dim=128,
        action_dim=7,
    )
    decoder = ActionDecoder(dec_config)

    # Pipeline
    pipeline = FullPipeline(encoder, decoder, freeze_encoder=False)

    img_prev = torch.rand(2, 3, 224, 224)
    img_curr = torch.rand(2, 3, 224, 224)

    output = pipeline(img_prev, img_curr)
    print(f"  Actions: {output['actions'].shape}")
    print(f"  Behavior embedding: {output['behavior_embedding'].shape}")

    # Loss test
    target_actions = torch.rand(2, 1, 7)
    loss = pipeline.compute_loss(img_prev, img_curr, target_actions)
    print(f"  Loss: {loss.item():.4f}")

    # Inference test
    action = pipeline.predict_action(img_prev[0], img_curr[0])
    print(f"  Single action: {action.shape}")

    print("\n✅ Full Pipeline tests passed!")
    return True


def test_memory_usage():
    """GPU 메모리 사용량 테스트"""
    print("\n" + "=" * 60)
    print("Testing Memory Usage")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("  ⚠️ CUDA not available, skipping")
        return True

    from src.models import TwoStreamInterleavedViT, TwoStreamViTConfig

    device = torch.device("cuda")

    config = TwoStreamViTConfig()  # Default config
    model = TwoStreamInterleavedViT(config).to(device)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    batch_size = 4
    img_prev = torch.rand(batch_size, 3, 224, 224, device=device)
    img_curr = torch.rand(batch_size, 3, 224, 224, device=device)

    with torch.cuda.amp.autocast():
        output = model(img_prev, img_curr)
        loss = output.mean()

    loss.backward()

    allocated = torch.cuda.memory_allocated() / 1e9
    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"  Batch size: {batch_size}")
    print(f"  Current GPU memory: {allocated:.2f} GB")
    print(f"  Peak GPU memory: {peak:.2f} GB")

    print("\n✅ Memory usage test passed!")
    return True


def main():
    print("\n" + "=" * 60)
    print("Two-Stream Interleaved ViT - Model Tests")
    print("=" * 60)

    tests = [
        ("Two-Stream Preprocessing", test_two_stream_preprocessing),
        ("Two-Stream Interleaved ViT", test_two_stream_interleaved_vit),
        ("Gradient Flow", test_gradient_flow),
        ("Action Decoder", test_action_decoder),
        ("Full Pipeline", test_full_pipeline),
        ("Memory Usage", test_memory_usage),
    ]

    results = []
    for name, test_fn in tests:
        try:
            success = test_fn()
            results.append((name, success))
        except Exception as e:
            print(f"\n❌ {name} FAILED: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    all_passed = True
    for name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"  {name}: {status}")
        all_passed = all_passed and success

    if all_passed:
        print("\n🎉 All tests passed!")
    else:
        print("\n⚠️ Some tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
