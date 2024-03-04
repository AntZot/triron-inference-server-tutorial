
import cv2
import tritonclient.http as httpclient


def main():
    client = httpclient.InferenceServerClient("localhost:8000")

    image = cv2.imread("")
    
    input_img = httpclient.InferInput("images",image.shape,"FP32")

    input_img.set_data_from_numpy(image,binary_data=True)

    response = client.infer("detection",input_img,"1")

    print(response.get_response())


if __name__ == "__main__":
    main()