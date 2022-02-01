#include "stdafx.h"
#include "common.h"
#include "Functions.h"
#include <queue>
using namespace std;
// Region growing pt modelul YCrCb cu seed point ales manual


void MyCallBackFunc_L4(int event, int x, int y, int flags, void* param)
{
	
	Mat* src = (Mat*)param;	

	Mat Cr = src[1];
	imshow("Cr component", Cr);

	Mat Cb = src[2];
	imshow("Cb component", Cb);
	
	if (event == CV_EVENT_RBUTTONDOWN) {

		//////////////////////////////////////////////////

		/// componenta Cr

		//Se seteaza eticheta curenta k = 1
		int k=1;
		//Se va aloca o matrice de etichete labels de dimensiunea imaginii.
		//Se initializeaza cu 0 fiecare element
		Mat labels = Mat::zeros(Cr.size(), CV_8UC3);
		//0 daca pixelul inca nu a fost parcurs, 1 daca pixelul a fost parcurs
		labels.at<uchar>(y, x)=k;

		//Se adauga elementul de start in lista FIFO
		// adauga element (seed point) in coada
		queue<Point> queCr;
		queCr.push(Point(x, y));

		//media ponderata pt componenta cr
		double cr_avg=0.0;
		cr_avg=Cr.at<uchar>(y, x);

		//numarul de pixeli din regiune
		int n = 1;
		
		while (!queCr.empty()) { //cat timp coada nu seste goala
			// Retine poz. celui mai vechi element din coada
			Point oldest = queCr.front();
			queCr.pop();// scoate element din coada
			int xx = oldest.x;// coordonatele lui
			int yy = oldest.y;
			//pentru fiecare vecin (i,j) al pixelului din pozitia „bottom” a listei
			for (int i = yy - 1; i <= yy + 1; i++) {
				for (int j = xx - 1; j <= xx + 1; j++) {
					// sunt in interiorul imaginii
					if (0<i && i<Cr.rows && 0<j && j <Cr.cols) {
						//pixel neprocesat inca si verifica conditia cu pragul 
						if ((abs(Cr.at<uchar>(i, j) - cr_avg) < 10) && (labels.at <uchar>(i, j) == 0)) {
							//adauga pixelul(i, j) in lista FIFO la pozitia top
							// Aduga vecin la regiunea curenta
							queCr.push(Point(j, i));
							//pixelul (i,j) primeste eticheta k: labels(i,j)=k
							labels.at<uchar>(i, j)=k;
							//se actualizeaza valoarea mediei ponderate a lui Cr
							cr_avg = ((n)*cr_avg + Cr.at<uchar>(i, j)) / (n + 1);
							//incrementeaza N
							n++;
						}
					}
				}
			}
		}
		Mat dstCr = Mat::zeros(Cr.size(), CV_8UC3);
		//Se vor afisa pixelii din regiunea curenta (labels(i,j) = 1) in imaginea destinatei 
		//cu alb (cei de fond vor fi negri )
		for (int i = 0; i < dstCr.rows; i++) {
			for (int j = 0; j < dstCr.cols; j++) {
				if (labels.at<uchar>(i, j) == 1) {
					dstCr.at<Vec3b>(i, j)[2] = 0;
					dstCr.at<Vec3b>(i, j)[1] = 0;
					dstCr.at<Vec3b>(i, j)[0] = 0;

				}
				else {
					dstCr.at<Vec3b>(i, j)[2] = 255;
					dstCr.at<Vec3b>(i, j)[1] = 255;
					dstCr.at<Vec3b>(i, j)[0] = 255;
				}
			}
		}
		//Imaginea segmentata poate prezenta zgomot
		//eliminam zgomotele
		//adica pixeli albi in interiorul obiectului si pixeli negrii in fundal
		//aplicam dilatari si eroziuni succesive 
		imshow("Region Grow- Cr", dstCr);
		Mat dstCr1 = dstCr.clone();
		Mat element1 = getStructuringElement(MORPH_CROSS, Size(3, 3));
		erode(dstCr1, dstCr1, element1, Point(-1, -1), 2);
		dilate(dstCr1, dstCr1, element1, Point(-1, -1), 4);
		erode(dstCr1, dstCr1, element1, Point(-1, -1), 2);
		imshow("PostProcesat CR", dstCr1);


		////////////////////////////////////////////////////

		// Componenta Cb
		//Se seteaza eticheta curenta kCb = 1 pt componenta Cb
		int kCb = 1; //ethiceta curenta 
		//Se va aloca o matrice de etichete labelsCb de dimensiunea imaginii Cb.
		// si se initializeaza cu 0 fiecare element
		Mat labelsCb = Mat::zeros(Cb.size(), CV_8UC3);
		labelsCb.at<uchar>(y, x) = kCb;

		//Se adauga elementul de start in lista FIFO
		queue<Point> queCb;
		queCb.push(Point(x, y));// adauga element (seed point) in coada
		
		//media ponderata pt componenta cb
		double cb_avg = 0.0;
		cb_avg = Cb.at<uchar>(y, x);

		int nCb = 1; // numarul de pixeli din regiune


		while (!queCb.empty()) {//cat timp coada nu e goala
			// Retin pozitia celui mai vechi element din coada
			Point oldest = queCb.front();
			queCb.pop(); // scoate element din coada
			int xx = oldest.x;// coordonata lui x
			int yy = oldest.y;// coordonata lui y
			// Pentru fiecare vecin al pixelului (xx, yy) ale carui coordonate
			for (int i = yy - 1; i <= yy + 1; i++) {
				for (int j = xx - 1; j <= xx + 1; j++) {
					// sunt in interiorul imaginii
					if (0 < i && i < Cr.rows && 0 < j && j < Cr.cols) {
						// Daca abs(cb(vecin) – cb_avg)<T si labels(vecin) == 0
						//pixel neprocesat inca si verifica conditia cu pragul 
						if ((abs(Cb.at<uchar>(i, j) - cb_avg) < 10) && (labelsCb.at <uchar>(i, j) == 0)) {
							// Aduga vecin la regiunea curenta
							queCb.push(Point(j, i));
							//pixelul (i,j) primeste eticheta k: labelsCb(i,j)=k
							// labels(vecin) = k
							labelsCb.at <uchar>(i, j) = kCb;
							// Actualizeaza cb_avg (medie ponderata)
							cb_avg = ((nCb)*cb_avg + Cb.at<uchar>(i, j)) / (nCb + 1);
							// Incrementeza N
							nCb++;
						}
					}
				}
			}
		}
		Mat dstCb = Mat::zeros(Cb.size(), CV_8UC3);
		//Se vor afisa pixelii din regiunea curenta (labels(i,j) = 1) in imaginea destinatei 
		//cu alb (cei de fond vor fi negri )
		for (int i = 0; i < dstCb.rows; i++) {
			for (int j = 0; j < dstCb.cols; j++) {
				if (labelsCb.at<uchar>(i, j) == 1) {
					dstCb.at<Vec3b>(i, j)[2] = 0;
					dstCb.at<Vec3b>(i, j)[1] = 0;
					dstCb.at<Vec3b>(i, j)[0] = 0;

				}
				else {
					dstCb.at<Vec3b>(i, j)[2] = 255;
					dstCb.at<Vec3b>(i, j)[1] = 255;
					dstCb.at<Vec3b>(i, j)[0] = 255;
				}
			}
		}
		imshow("Region Grow- Cb", dstCb);
		Mat dstCb1 = dstCb.clone();		
		erode(dstCb1, dstCb1, element1, Point(-1, -1), 2);
		dilate(dstCb1, dstCb1, element1, Point(-1, -1), 4);
		erode(dstCb1, dstCb1, element1, Point(-1, -1), 2);
		imshow("PostProcesat CB", dstCb1);

		////////////////////////////////////////////////////

		// Componenta Cr+Cb
		
		//distanta dintre componentele Cb si Cr
		double d = 0.0;
		double d1=0.0;
		//Se va aloca o matrice de etichete labels de dimensiunea imaginii.
		//Se initializeaza cu 0 fiecare element
		Mat labelsComb = Mat::zeros(Cb.size(), CV_8UC3);
		//0 daca pixelul inca nu a fost parcurs, 1 daca pixelul a fost parcurs
		labelsComb.at<uchar>(y, x) = k;
		//Se adauga elementul de start in lista FIFO
		// adauga element (seed point) in coada
		queue<Point> queComb;
		queComb.push(Point(x, y));
		//cat timp coada nu seste goala
		while (!queComb.empty()){
			// Retine poz. celui mai vechi element din coada
			Point oldest = queComb.front();
			// scoate element din coada
			queComb.pop();
			int xx = oldest.x; // coordonata lui x 
			int yy = oldest.y;// coordonata lui y
			//pentru fiecare vecin (i,j) al pixelului din pozitia „bottom” a listei
			for (int i = yy - 3; i <= yy + 3; i++) {
				for (int j = xx - 3; j <= xx + 3; j++) {
					// sunt in interiorul imaginii
					if (0 < i && i < Cr.rows && 0 < j && j < Cr.cols) {
						//calculez distnata dintre cele doua componente cb si cr
						d1 = pow((Cb.at<uchar>(i, j) - cb_avg),2) + pow((Cr.at<uchar>(i, j) - cr_avg),2);
						d = sqrt(d1);
						//pixel neprocesat inca si verifica conditia cu pragul pt distanta
						if (d <10 && labelsComb.at<uchar>(i, j) == 0) {
							//adauga pixelul(i, j) in lista FIFO la pozitia top
							// Aduga vecin la regiunea curenta
							queComb.push(Point(j, i));
							//pixelul (i,j) primeste eticheta k: labelsComb(i,j)=k
							labelsComb.at <uchar>(i, j) = k;
							//se actualizeaza valoarea mediei ponderate a lui Cr si Cb
							cr_avg = ((n)*cr_avg + Cr.at<uchar>(i, j)) / (n + 1);
							cb_avg = ((n)*cb_avg + Cb.at<uchar>(i, j)) / (n + 1);
							n++;
						}
					}
				}
			}

		}


		Mat dstCbCr = Mat::zeros(Cb.size(), CV_8UC3);
		//Se vor afisa pixelii din regiunea curenta (labels(i,j) = 1) in imaginea destinatei 
		//cu alb (cei de fond vor fi negri )
		for (int i = 0; i < dstCbCr.rows; i++) {
			for (int j = 0; j < dstCbCr.cols; j++) {
				if (labelsComb.at<uchar>(i, j) == 1) {
					dstCbCr.at<Vec3b>(i, j)[0] = 0;
					dstCbCr.at<Vec3b>(i, j)[1] = 0;
					dstCbCr.at<Vec3b>(i, j)[2] = 0;

				}
				else {
					dstCbCr.at<Vec3b>(i, j)[2] = 255;
					dstCbCr.at<Vec3b>(i, j)[1] = 255;
					dstCbCr.at<Vec3b>(i, j)[0] = 255;
				}
			}
		}
		//Imaginea segmentata poate prezenta zgomot
		//eliminam zgomotele
		//adica pixeli albi in interiorul obiectului si pixeli negrii in fundal
		//aplicam dilatari si eroziuni succesive 
		imshow("Region Grow- CbCr", dstCbCr);
		Mat dstCbCr1 = dstCbCr.clone();
		erode(dstCbCr1, dstCbCr1, element1, Point(-1, -1), 2);
		dilate(dstCbCr1, dstCbCr1, element1, Point(-1, -1), 4);
		erode(dstCbCr1, dstCbCr1, element1, Point(-1, -1), 2);
		imshow("PostProcesat CBCR", dstCbCr1);
		
	}
}


void conversie(){	
	char fname[MAX_PATH];
	while (openFileDlg(fname))	{
		Mat src = imread(fname);
		imshow("src", src);
		int height = src.rows;
		int width = src.cols;
		Mat src2= Mat(height, width, CV_8UC3);		
		//conversie BGR -> YCrCb
		cvtColor(src, src2, COLOR_BGR2YCrCb);
		
		Mat different_Channels[3];
		split(src2, different_Channels);//impart imaginea in 3 canale  

		Mat y = different_Channels[0]; //matricea pt y
		Mat cr = different_Channels[1];//matricea pt cr
		Mat cb = different_Channels[2];  //matricea pt cb 

		//afisez cele 3 matrici de culoare 
		imshow("y", y);
		imshow("cr", cr);
		imshow("cb", cb);
		
		imshow("Imagine YCbCr", src2);
		setMouseCallback("src", MyCallBackFunc_L4, &different_Channels);
		waitKey(0);
	}
	
}


int main()
{
	int op;
	do
	{
		system("cls");
		destroyAllWindows();
		printf("Menu:\n");
		printf(" 1 - Region Growing Cr - Cb - CrCb \n");
		printf(" 0 - Exit\n\n");
		printf("Option: ");
		scanf("%d",&op);
		switch (op)
		{
			case 1 :
			conversie() ;
			break;
		}
	}
	while (op!=0);
	return 0;
}