#!/usr/bin/env python3
"""
Demo script for testing the complete ML pipeline integration.

This script demonstrates the CaptionService functionality with various
image samples and validates the output quality of the integrated system.
"""

import os
import sys
import time
import logging
from io import BytesIO
from PIL import Image
import numpy as np

# Add the app directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from app.services.caption_service import CaptionService, CaptionResult

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_test_images():
    """
    Create diverse test images for pipeline validation.
    
    Returns:
        Dict[str, bytes]: Dictionary of image name to image data
    """
    test_images = {}
    
    # Simple solid color image
    img = Image.new('RGB', (300, 200), color='red')
    buffer = BytesIO()
    img.save(buffer, format='JPEG')
    test_images['red_rectangle'] = buffer.getvalue()
    
    # Gradient image
    img = Image.new('RGB', (400, 300), color='white')
    pixels = img.load()
    for i in range(img.width):
        for j in range(img.height):
            # Create a blue to green gradient
            blue_val = int(255 * (1 - i / img.width))
            green_val = int(255 * (i / img.width))
            pixels[i, j] = (0, green_val, blue_val)
    
    buffer = BytesIO()
    img.save(buffer, format='JPEG')
    test_images['blue_green_gradient'] = buffer.getvalue()
    
    # Pattern image (checkerboard)
    img = Image.new('RGB', (200, 200), color='white')
    pixels = img.load()
    square_size = 20
    for i in range(img.width):
        for j in range(img.height):
            if ((i // square_size) + (j // square_size)) % 2 == 0:
                pixels[i, j] = (0, 0, 0)  # Black
            else:
                pixels[i, j] = (255, 255, 255)  # White
    
    buffer = BytesIO()
    img.save(buffer, format='PNG')
    test_images['checkerboard_pattern'] = buffer.getvalue()
    
    # Circular pattern
    img = Image.new('RGB', (250, 250), color='white')
    pixels = img.load()
    center_x, center_y = img.width // 2, img.height // 2
    for i in range(img.width):
        for j in range(img.height):
            distance = ((i - center_x) ** 2 + (j - center_y) ** 2) ** 0.5
            if distance < 50:
                pixels[i, j] = (255, 0, 0)  # Red center
            elif distance < 100:
                pixels[i, j] = (0, 255, 0)  # Green ring
            else:
                pixels[i, j] = (0, 0, 255)  # Blue outer
    
    buffer = BytesIO()
    img.save(buffer, format='WEBP')
    test_images['concentric_circles'] = buffer.getvalue()
    
    # Random noise image
    noise_array = np.random.randint(0, 256, (150, 150, 3), dtype=np.uint8)
    img = Image.fromarray(noise_array)
    buffer = BytesIO()
    img.save(buffer, format='JPEG')
    test_images['random_noise'] = buffer.getvalue()
    
    return test_images


def test_caption_service_initialization():
    """Test CaptionService initialization."""
    logger.info("Testing CaptionService initialization...")
    
    service = CaptionService(
        target_image_size=(224, 224),
        max_file_size=10 * 1024 * 1024,
        default_temperature=1.0,
        use_beam_search=False
    )
    
    # Test initialization
    start_time = time.time()
    success = service.initialize()
    init_time = time.time() - start_time
    
    logger.info(f"Initialization {'successful' if success else 'failed'} in {init_time:.2f}s")
    
    if not success:
        logger.error(f"Initialization error: {service._initialization_error}")
        return False
    
    # Test service status
    status = service.get_service_status()
    logger.info(f"Service status: {status}")
    
    return True


def test_image_validation(service, test_images):
    """Test image validation functionality."""
    logger.info("Testing image validation...")
    
    for name, image_data in test_images.items():
        is_valid, error = service.validate_image_quick(image_data)
        logger.info(f"Image '{name}': {'Valid' if is_valid else f'Invalid - {error}'}")
        
        if not is_valid:
            logger.warning(f"Image validation failed for {name}: {error}")


def test_caption_generation_modes(service, test_images):
    """Test different caption generation modes."""
    logger.info("Testing caption generation modes...")
    
    # Test with first image using different modes
    test_image_name = list(test_images.keys())[0]
    test_image_data = test_images[test_image_name]
    
    logger.info(f"Testing with image: {test_image_name}")
    
    # Test greedy decoding
    logger.info("Testing greedy decoding...")
    result_greedy = service.generate_caption(
        test_image_data,
        temperature=1.0,
        use_beam_search=False
    )
    
    if result_greedy.success:
        logger.info(f"Greedy result: '{result_greedy.caption}' (confidence: {result_greedy.confidence:.3f})")
    else:
        logger.error(f"Greedy generation failed: {result_greedy.error_message}")
    
    # Test beam search
    logger.info("Testing beam search...")
    result_beam = service.generate_caption(
        test_image_data,
        temperature=1.0,
        use_beam_search=True,
        beam_width=3
    )
    
    if result_beam.success:
        logger.info(f"Beam search result: '{result_beam.caption}' (confidence: {result_beam.confidence:.3f})")
    else:
        logger.error(f"Beam search generation failed: {result_beam.error_message}")
    
    # Test different temperatures
    for temp in [0.5, 1.0, 1.5]:
        logger.info(f"Testing temperature {temp}...")
        result = service.generate_caption(
            test_image_data,
            temperature=temp,
            use_beam_search=False
        )
        
        if result.success:
            logger.info(f"Temperature {temp}: '{result.caption}' (confidence: {result.confidence:.3f})")
        else:
            logger.error(f"Temperature {temp} failed: {result.error_message}")


def test_diverse_images(service, test_images):
    """Test caption generation with diverse image samples."""
    logger.info("Testing caption generation with diverse images...")
    
    results = []
    
    for name, image_data in test_images.items():
        logger.info(f"Processing image: {name}")
        
        start_time = time.time()
        result = service.generate_caption(image_data)
        processing_time = time.time() - start_time
        
        if result.success:
            logger.info(f"  Caption: '{result.caption}'")
            logger.info(f"  Confidence: {result.confidence:.3f}")
            logger.info(f"  Processing time: {result.processing_time:.3f}s")
            logger.info(f"  Image metadata: {result.image_metadata}")
            
            results.append({
                'name': name,
                'caption': result.caption,
                'confidence': result.confidence,
                'processing_time': result.processing_time,
                'success': True
            })
        else:
            logger.error(f"  Failed: {result.error_message}")
            results.append({
                'name': name,
                'error': result.error_message,
                'success': False
            })
        
        logger.info("")
    
    return results


def test_error_scenarios(service):
    """Test error handling scenarios."""
    logger.info("Testing error handling scenarios...")
    
    # Test with invalid image data
    logger.info("Testing with invalid image data...")
    result = service.generate_caption(b"not an image")
    if not result.success:
        logger.info(f"  Correctly handled invalid data: {result.error_message}")
    else:
        logger.warning("  Should have failed with invalid data")
    
    # Test with empty data
    logger.info("Testing with empty data...")
    result = service.generate_caption(b"")
    if not result.success:
        logger.info(f"  Correctly handled empty data: {result.error_message}")
    else:
        logger.warning("  Should have failed with empty data")
    
    # Test with oversized image (create a large image)
    logger.info("Testing with oversized image...")
    large_img = Image.new('RGB', (5000, 5000), color='blue')
    buffer = BytesIO()
    large_img.save(buffer, format='JPEG', quality=95)
    large_data = buffer.getvalue()
    
    # Create service with small size limit
    small_service = CaptionService(max_file_size=1000)  # 1KB limit
    result = small_service.generate_caption(large_data)
    if not result.success:
        logger.info(f"  Correctly handled oversized image: {result.error_message}")
    else:
        logger.warning("  Should have failed with oversized image")


def test_performance_benchmarks(service, test_images):
    """Test performance benchmarks."""
    logger.info("Running performance benchmarks...")
    
    # Warm up the service
    first_image = list(test_images.values())[0]
    service.generate_caption(first_image)
    
    # Benchmark multiple requests
    num_requests = 5
    total_time = 0
    successful_requests = 0
    
    for i in range(num_requests):
        image_data = list(test_images.values())[i % len(test_images)]
        
        start_time = time.time()
        result = service.generate_caption(image_data)
        request_time = time.time() - start_time
        
        if result.success:
            successful_requests += 1
            total_time += request_time
            logger.info(f"  Request {i+1}: {request_time:.3f}s - '{result.caption}'")
        else:
            logger.error(f"  Request {i+1} failed: {result.error_message}")
    
    if successful_requests > 0:
        avg_time = total_time / successful_requests
        logger.info(f"Average processing time: {avg_time:.3f}s ({successful_requests}/{num_requests} successful)")
    else:
        logger.error("No successful requests in benchmark")


def validate_output_quality(results):
    """Validate the quality of generated captions."""
    logger.info("Validating output quality...")
    
    successful_results = [r for r in results if r['success']]
    
    if not successful_results:
        logger.error("No successful results to validate")
        return False
    
    # Check that captions are non-empty
    empty_captions = [r for r in successful_results if not r['caption'].strip()]
    if empty_captions:
        logger.warning(f"Found {len(empty_captions)} empty captions")
    
    # Check caption lengths
    caption_lengths = [len(r['caption']) for r in successful_results]
    avg_length = sum(caption_lengths) / len(caption_lengths)
    logger.info(f"Average caption length: {avg_length:.1f} characters")
    
    # Check confidence scores
    confidences = [r['confidence'] for r in successful_results]
    avg_confidence = sum(confidences) / len(confidences)
    logger.info(f"Average confidence: {avg_confidence:.3f}")
    
    # Check processing times
    processing_times = [r['processing_time'] for r in successful_results]
    avg_processing_time = sum(processing_times) / len(processing_times)
    logger.info(f"Average processing time: {avg_processing_time:.3f}s")
    
    # Basic quality checks
    quality_issues = []
    
    for result in successful_results:
        caption = result['caption']
        
        # Check for reasonable caption structure
        if not caption.startswith(('a ', 'an ', 'the ')):
            quality_issues.append(f"Caption doesn't start with article: '{caption}'")
        
        # Check for reasonable length
        if len(caption) < 5:
            quality_issues.append(f"Caption too short: '{caption}'")
        
        # Check confidence is reasonable
        if result['confidence'] < 0.01:
            quality_issues.append(f"Very low confidence ({result['confidence']:.3f}): '{caption}'")
    
    if quality_issues:
        logger.warning(f"Found {len(quality_issues)} quality issues:")
        for issue in quality_issues[:5]:  # Show first 5 issues
            logger.warning(f"  {issue}")
    else:
        logger.info("No major quality issues detected")
    
    return len(quality_issues) == 0


def main():
    """Main demo function."""
    logger.info("Starting complete ML pipeline demo...")
    
    try:
        # Create test images
        logger.info("Creating test images...")
        test_images = create_test_images()
        logger.info(f"Created {len(test_images)} test images")
        
        # Initialize service
        service = CaptionService()
        if not test_caption_service_initialization():
            logger.error("Service initialization failed, aborting demo")
            return False
        
        # Test image validation
        test_image_validation(service, test_images)
        
        # Test different caption generation modes
        test_caption_generation_modes(service, test_images)
        
        # Test with diverse images
        results = test_diverse_images(service, test_images)
        
        # Test error scenarios
        test_error_scenarios(service)
        
        # Performance benchmarks
        test_performance_benchmarks(service, test_images)
        
        # Validate output quality
        quality_ok = validate_output_quality(results)
        
        # Summary
        successful_results = [r for r in results if r['success']]
        logger.info(f"Demo completed: {len(successful_results)}/{len(results)} images processed successfully")
        
        if quality_ok:
            logger.info("Output quality validation passed")
        else:
            logger.warning("Output quality validation found issues")
        
        return True
        
    except Exception as e:
        logger.error(f"Demo failed with exception: {str(e)}", exc_info=True)
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)