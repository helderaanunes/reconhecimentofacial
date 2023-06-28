package br.com.haanunes.reconhecimentofacial;

import java.awt.event.KeyEvent;
import java.util.logging.Level;
import java.util.logging.Logger;
import javax.swing.JOptionPane;
import org.bytedeco.javacpp.DoublePointer;
import org.bytedeco.javacpp.IntPointer;
import org.bytedeco.javacv.CanvasFrame;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.FrameGrabber;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.bytedeco.javacv.OpenCVFrameGrabber;
import org.bytedeco.opencv.global.opencv_imgcodecs;
import org.bytedeco.opencv.global.opencv_imgproc;
import static org.bytedeco.opencv.global.opencv_imgproc.resize;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Point;
import org.bytedeco.opencv.opencv_core.Rect;
import org.bytedeco.opencv.opencv_core.RectVector;
import org.bytedeco.opencv.opencv_core.Scalar;
import org.bytedeco.opencv.opencv_core.Size;
import org.bytedeco.opencv.opencv_face.EigenFaceRecognizer;
import org.bytedeco.opencv.opencv_face.FaceRecognizer;
import org.bytedeco.opencv.opencv_objdetect.CascadeClassifier;
import org.opencv.imgproc.Imgproc;

/**
 * @author Helder
 */
public class Reconhecimento {

    public static void main(String[] args) {
        try {
            //convertMat, vai mapear (ligar) os dados de map entre o frame e a imagem
            OpenCVFrameConverter.ToMat convertMat = new OpenCVFrameConverter.ToMat();
            // pega nosso dispositivo de acordo com o parâmetro. nesse caso o 0 (zero) é o primeiro
            // dispositivo de câmera do computador (um computador pode ter várias câmeras conectadas
            OpenCVFrameGrabber camera = new OpenCVFrameGrabber(0);
            camera.start();
            CascadeClassifier detectorFace = new CascadeClassifier("src/main/java/recursos/haarcascade-frontalface.xml");
            
            FaceRecognizer reconhecedor = EigenFaceRecognizer.create();
            reconhecedor.read("src/main/java/recursos/eigen.yml");
            String [] pessoas = {"", "Helder","Miguel"};
            
            Mat imagemColorida = new Mat();

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

                for (int i = 0; i < facesDetectadas.size(); i++) {
                    Rect dadosFace = facesDetectadas.get(i);
                    opencv_imgproc.rectangle(imagemColorida, dadosFace, new Scalar(0, 255, 0, 0));

                    Mat faceCapturada = new Mat(imagemCinza, dadosFace);
                    resize(faceCapturada, faceCapturada, new Size(160, 160));
                    
                    IntPointer rotuloEtiqueta = new IntPointer(1);
                    DoublePointer confianca = new DoublePointer(1);
                    reconhecedor.predict(faceCapturada, rotuloEtiqueta, confianca);
                    int predicao = rotuloEtiqueta.get(0);
                    String nomePessoa;
                    if (predicao== -1){ // não achou ninguem
                        nomePessoa="Desconhecido";
                    }
                    else{
                        nomePessoa=pessoas[predicao]+ " - "+confianca.get(0);
                    }
                    int x = Math.max(dadosFace.tl().x()-10, 0);
                    int y = Math.max(dadosFace.tl().y()-10, 0);
                    //imagem, texto, posição, fonte do texto, tamanho da fonte, cor
                    opencv_imgproc.putText(imagemColorida,nomePessoa,new Point(x,y), Imgproc.FONT_HERSHEY_DUPLEX,1.4,new Scalar(0,255,0,0));
                }
                if (cFrame.isVisible()) {
                    cFrame.showImage(frameCapturado);
                }

            }
            // para encerrar a janela e a camera (economizar memória e processamento) quando sair do laço.
            cFrame.dispose();
            camera.stop();

        } catch (FrameGrabber.Exception ex) {
            Logger.getLogger(Reconhecimento.class.getName()).log(Level.SEVERE, null, ex);
        } 
    }
}
