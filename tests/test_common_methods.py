import pytest
from potholeClassifier.utils.common import *
from potholeClassifier.constants import *
from box import ConfigBox
import io
import tempfile
import os


def test_read_yaml_valid_file() -> None:
    """
    Test that the read_yaml function correctly reads a valid YAML file and
    returns an instance of the ConfigBox class.

    Steps:
    1. Call read_yaml with CONFIG_FILE_PATH.
    2. Assert that the result is an instance of ConfigBox.

    Expected Outcome:
    The function should return an object of type ConfigBox.

    """
    result = read_yaml(Path('../config/config.yaml'))
    assert isinstance(result, ConfigBox)


def test_read_yaml_nonexistent_file() -> None:
    """
    Test that the read_yaml function raises a FileNotFoundError when provided
    with a path to a non-existent file.

    Steps:
    1. Use pytest.raises to catch a FileNotFoundError.
    2. Call read_yaml with the EMPTY_FILE_PATH.

    Expected Outcome:
    The function should raise a FileNotFoundError.

    """
    with pytest.raises(FileNotFoundError):
        read_yaml(Path(EMPTY_FILE_PATH))


def test_encodeImageIntoBase64():
    """
    Tests the encodeImageIntoBase64 function to ensure it accurately converts
    an image to a Base64 string and that this string can be decoded back to an
    image with the same properties as the original.

    Steps:
    1. Create an RGB image of size 100x100.
    2. Save the image into a temporary file in JPEG format.
    3. Encode the image into a Base64 string.
    4. Decode the Base64 string back into bytes.
    5. Load the image from the decoded bytes.
    6. Verify that the mode and size of the decoded image match the original image.
    7. Close the decoded image.

    Expected Outcome:
    The decoded image should have the same mode ('RGB') and size (100x100) as the
    original image. If these properties match, the encodeImageIntoBase64 function 
    works correctly. If not, the function has a bug.
    """
 
    img = Image.new("RGB", (100, 100))

    with tempfile.NamedTemporaryFile(suffix=".jpg") as tmp_file:
        img.save(tmp_file, format="JPEG")
        tmp_file.flush()  

        img_b64 = encodeImageIntoBase64(tmp_file.name)

    decoded_img_bytes = base64.b64decode(img_b64)

    decoded_img = Image.open(io.BytesIO(decoded_img_bytes))

    assert decoded_img.mode == img.mode
    assert decoded_img.size == img.size

    decoded_img.close()




def test_decodeImage():
    """
    Tests the decodeImage function to ensure it accurately decodes a Base64 string
    and saves it as an image file with the same properties as the original image.

    Steps:
    1. Create an RGB image of size 100x100.
    2. Save the image into a BytesIO object in JPEG format.
    3. Encode the image into a Base64 string.
    4. Decode the Base64 string back into bytes.
    5. Load the image from the decoded bytes.
    6. Test the decodeImage function by saving the decoded image to a file.
    7. Verify that the mode and size of the saved image match the decoded image.
    8. Clean up the created image files and objects.

    Expected Outcome:
    The image saved by the decodeImage function should have the same mode ('RGB')
    and size (100x100) as the original image. If these properties match, the
    decodeImage function works correctly. If not, the function has a bug.

    """
   
    img = Image.new("RGB", (100, 100))
    
    img_byte_array = BytesIO()
    img.save(img_byte_array, format="JPEG")
    img_byte_array.seek(0)
    img_bytes = img_byte_array.getvalue()
    
    img_b64 = base64.b64encode(img_bytes).decode()
    
    decoded_img_bytes = base64.b64decode(img_b64)
    
    decoded_img = Image.open(BytesIO(decoded_img_bytes))
    
    decodeImage(img_b64, "test.jpg")
    test_img = Image.open("test.jpg")
    
    assert test_img.mode == decoded_img.mode, "Image mode does not match"
    assert test_img.size == decoded_img.size, "Image size does not match"
    
    test_img.close()
    decoded_img.close()
    os.remove("test.jpg")