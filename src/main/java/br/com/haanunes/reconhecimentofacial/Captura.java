package br.com.haanunes.reconhecimentofacial;

import java.awt.event.KeyEvent;
import java.util.logging.Level;
import java.util.logging.Logger;
import javax.swing.JOptionPane;
import org.bytedeco.javacv.CanvasFrame;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.FrameGrabber;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.bytedeco.javacv.OpenCVFrameGrabber;
import org.bytedeco.opencv.global.opencv_cudaimgproc;
import org.bytedeco.opencv.global.opencv_imgcodecs;
import org.bytedeco.opencv.global.opencv_imgproc;
import static org.bytedeco.opencv.global.opencv_imgproc.resize;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Rect;
import org.bytedeco.opencv.opencv_core.RectVector;
import org.bytedeco.opencv.opencv_core.Scalar;
import org.bytedeco.opencv.opencv_core.Size;
import org.bytedeco.opencv.opencv_objdetect.CascadeClassifier;

/**
 * @author Helder
 */
public class Captura {

    public static void main(String[] args) {
        try {
            KeyEvent tecla = null;

            //convertMat, vai mapear (ligar) os dados de map entre o frame e a imagem
            OpenCVFrameConverter.ToMat convertMat = new OpenCVFrameConverter.ToMat();
            // pega nosso dispositivo de acordo com o parâmetro. nesse caso o 0 (zero) é o primeiro
            // dispositivo de câmera do computador (um computador pode ter várias câmeras conectadas
            OpenCVFrameGrabber camera = new OpenCVFrameGrabber(0);
            camera.start();
            CascadeClassifier detectorFace = new CascadeClassifier("src/main/java/recursos/haarcascade-frontalface.xml");
            Mat imagemColorida = new Mat();

            int numeroAmostrarTreinamento = 20;
            int contador = 1;
            int id = Integer.parseInt(JOptionPane.showInputDialog(null, "Digite o id da pessoa:"));
            // cria um quadro (janela) para colocar a imagem da câmera
            CanvasFrame cFrame = new CanvasFrame("Nossa câmera", CanvasFrame.getDefaultGamma() / camera.getGamma());
            cFrame.setDefaultCloseOperation(CanvasFrame.EXIT_ON_CLOSE);
            //fica atualizando a imagem da câmera enquanto tiver novas imagens
            Frame frameCapturado = null;
            while ((frameCapturado = camera.grab()) != null) {

                imagemColorida = convertMat.convert(frameCapturado);
                Mat imagemCinza = new Mat();
                opencv_imgproc.cvtColor(imagemColorida, imagemCinza, opencv_imgproc.COLOR_BGRA2GRAY);
                RectVector facesDetectadas = new RectVector();
                detectorFace.detectMultiScale(imagemCinza, facesDetectadas, 1.1, 1, 0, new Size(150, 150), new Size(500, 500));

                if (tecla == null) {
                    tecla = cFrame.waitKey(5);
                }

                for (int i = 0; i < facesDetectadas.size(); i++) {
                    Rect dadosFace = facesDetectadas.get(0);
                    opencv_imgproc.rectangle(imagemColorida, dadosFace, new Scalar(0, 255, 0, 0));

                    Mat faceCapturada = new Mat(imagemCinza, dadosFace);
                    resize(faceCapturada, faceCapturada, new Size(160, 160));
                    if (tecla == null) {
                        tecla = cFrame.waitKey(5);
                    }
                    if (tecla != null) {
                        if (tecla.getKeyChar() == 'f') {
                            opencv_imgcodecs.imwrite("src/main/java/fotos/" + id + "_" + contador + ".jpg", faceCapturada);
                            System.out.println("foto " + contador);
                            contador++;
                        }
                        tecla = null;
                    }
                }
                if (tecla == null) {
                    tecla = cFrame.waitKey(20);
                }
                if (contador > 25) {
                    break;
                }
                if (cFrame.isVisible()) {
                    cFrame.showImage(frameCapturado);
                }

            }
            // para encerrar a janela e a camera (economizar memória e processamento) quando sair do laço.
            cFrame.dispose();
            camera.stop();

        } catch (FrameGrabber.Exception ex) {
            Logger.getLogger(Captura.class.getName()).log(Level.SEVERE, null, ex);
        } catch (InterruptedException ex) {
            Logger.getLogger(Captura.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
}
