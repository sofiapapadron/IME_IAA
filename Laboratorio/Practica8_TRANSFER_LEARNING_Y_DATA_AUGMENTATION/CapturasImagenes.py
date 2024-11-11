import cv2
import os

n = 0

general_path = "Data"
training_path = "training"
validation_path = "validation"
uno_path = "1"
dos_path = "2"
tres_path = "3"
cuatro_path = "4"

width = 24
height = 224

def Carpeta(image_dir_path):
    CHECK_DIR = os.path.isdir(image_dir_path)
    if not CHECK_DIR:
        os.makedirs(image_dir_path)
        print(f'"{image_dir_path}" directorio ha sido creado')
    #else:
        #print(f'"{image_dir_path}" directorio ya existe')

def CapturaImagenes(cap, toma_path):
     global n
     for i in range(100):
        _, frame = cap.read()
        copyFrame = frame.copy()

        cv2.imshow("Original", copyFrame)

        if i <= 80:
            Carpeta(general_path + '/' + training_path + '/' + toma_path)
            copyFrame = cv2.resize(copyFrame, (width, height))
            copyFrame = cv2.cvtColor(copyFrame, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(f"{general_path + '/' + training_path + '/' + toma_path}/image{n}.png", copyFrame)
            print(f"Numero de imagen: #{n}")
            n += 1
        else:
            Carpeta(general_path + '/' + validation_path + '/' + toma_path)
            copyFrame = cv2.resize(copyFrame, (width, height))
            copyFrame = cv2.cvtColor(copyFrame, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(f"{general_path + '/' + validation_path + '/' + toma_path}/image{n}.png", copyFrame)
            print(f"Numero de imagen: #{n}")
            n += 1
        cv2.waitKey(50)

def main():    

    Carpeta(general_path)
    Carpeta(general_path + '/' + training_path)
    Carpeta(general_path + '/' + validation_path)

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    while True:
        _, frame = cap.read()
        copyFrame = frame.copy()

        cv2.imshow("Original", copyFrame)

        key = cv2.waitKey(1)

        if key == ord("q"):
            break
        if key == ord("a"):
            CapturaImagenes(cap, uno_path)
        if key == ord("b"):
            CapturaImagenes(cap, dos_path)
        if key == ord("c"):
            CapturaImagenes(cap, tres_path)
        if key == ord("d"):
            CapturaImagenes(cap, cuatro_path)


    cap.release()
    cv2.destroyAllWindows()

    print("Total de imagenes guardadas:", n)

if __name__ == "__main__":
    main() 