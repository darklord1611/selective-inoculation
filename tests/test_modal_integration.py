"""Quick test to verify Modal serving integration.

This script tests that all modules import correctly and basic functionality works.
"""
import sys


def test_imports():
    """Test that all Modal serving modules import correctly."""
    print("Testing imports...")

    try:
        from mi.modal_serving import (
            ModalServingConfig,
            ModalEndpoint,
            deploy_endpoint,
            get_or_deploy_endpoint,
            list_endpoints,
        )
        print("✓ mi.modal_serving imports OK")
    except ImportError as e:
        print(f"✗ Failed to import mi.modal_serving: {e}")
        return False

    try:
        from mi.modal_serving import modal_app
        print("✓ mi.modal_serving.modal_app imports OK")
    except ImportError as e:
        print(f"✗ Failed to import mi.modal_serving.modal_app: {e}")
        return False

    try:
        from mi.external.modal_driver import services as modal_services
        print("✓ mi.external.modal_driver imports OK")
    except ImportError as e:
        print(f"✗ Failed to import mi.external.modal_driver: {e}")
        return False

    try:
        from mi.llm.data_models import Model
        from mi.llm.services import sample, batch_sample
        print("✓ mi.llm updated imports OK")
    except ImportError as e:
        print(f"✗ Failed to import updated mi.llm: {e}")
        return False

    return True


def test_data_models():
    """Test that data models can be instantiated."""
    print("\nTesting data models...")

    try:
        from mi.modal_serving import ModalServingConfig, ModalEndpoint
        from mi.llm.data_models import Model

        # Test ModalServingConfig
        config = ModalServingConfig(
            base_model_id="Qwen/Qwen2.5-1.5B-Instruct",
            lora_path="/training_out/test",
            lora_name="test-adapter",
        )
        print(f"✓ ModalServingConfig: {config.base_model_id}")

        # Test ModalEndpoint
        endpoint = ModalEndpoint(
            config=config,
            endpoint_url="https://test.modal.run/v1",
            app_name="test-app",
        )
        print(f"✓ ModalEndpoint: {endpoint.endpoint_url}")

        # Test Model with modal type
        model = Model(
            id="test-model",
            type="modal",
            modal_endpoint_url="https://test.modal.run/v1",
            modal_api_key="test-key",
        )
        print(f"✓ Model (modal type): {model.id}")

        return True
    except Exception as e:
        print(f"✗ Failed to instantiate data models: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_modal_app_creation():
    """Test that Modal app structure is correct."""
    print("\nTesting Modal app structure...")

    try:
        from mi.modal_serving import modal_app

        # Verify the app exists
        if not hasattr(modal_app, 'app'):
            print("✗ modal_app.app does not exist")
            return False
        print("✓ modal_app.app exists")

        # Verify the serve function exists
        if not hasattr(modal_app, 'serve'):
            print("✗ modal_app.serve function does not exist")
            return False
        print("✓ modal_app.serve function exists")

        # Verify app is a Modal app
        import modal
        if not isinstance(modal_app.app, modal.App):
            print("✗ modal_app.app is not a Modal App instance")
            return False
        print("✓ modal_app.app is a Modal App instance")

        print("  (Functions are defined at module top-level to avoid nesting issues)")

        return True
    except Exception as e:
        print(f"✗ Failed to verify Modal app structure: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_endpoint_management():
    """Test endpoint caching and listing."""
    print("\nTesting endpoint management...")

    try:
        from mi.modal_serving import list_endpoints

        endpoints = list_endpoints()
        print(f"✓ Found {len(endpoints)} cached endpoints")

        return True
    except Exception as e:
        print(f"✗ Failed endpoint management test: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("Modal Serving Integration Tests")
    print("=" * 60)

    tests = [
        ("Imports", test_imports),
        ("Data Models", test_data_models),
        ("Modal App Creation", test_modal_app_creation),
        ("Endpoint Management", test_endpoint_management),
    ]

    results = []
    for name, test_func in tests:
        print(f"\n{'=' * 60}")
        print(f"Test: {name}")
        print('=' * 60)
        passed = test_func()
        results.append((name, passed))

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")

    all_passed = all(passed for _, passed in results)
    print("=" * 60)

    if all_passed:
        print("\n✓ All tests passed!")
        return 0
    else:
        print("\n✗ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
