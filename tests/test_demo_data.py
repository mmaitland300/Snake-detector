from snake_detector.demo_data import generate_demo_dataset


def test_generate_demo_dataset_creates_expected_class_dirs(tmp_path) -> None:
    output_dir = tmp_path / "public_demo_raw"
    generate_demo_dataset(output_dir, samples_per_class=3, image_size=64, seed=7)

    snake_files = sorted((output_dir / "snake").glob("*.png"))
    no_snake_files = sorted((output_dir / "no_snake").glob("*.png"))

    assert len(snake_files) == 3
    assert len(no_snake_files) == 3
