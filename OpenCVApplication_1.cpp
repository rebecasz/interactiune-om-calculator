// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include "Functions.h"

#define MAX_HUE 256
int histc_hue[MAX_HUE];
#define MAX_HUE 256
//variabile globale
int histG_hue[MAX_HUE]; // histograma globala / cumulativa

#include <queue>
using namespace std;

void testOpenImage()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		imshow("image",src);
		waitKey();
	}
}

void testOpenImagesFld()
{
	char folderName[MAX_PATH];
	if (openFolderDlg(folderName)==0)
		return;
	char fname[MAX_PATH];
	FileGetter fg(folderName,"bmp");
	while(fg.getNextAbsFile(fname))
	{
		Mat src;
		src = imread(fname);
		imshow(fg.getFoundFileName(),src);
		if (waitKey()==27) //ESC pressed
			break;
	}
}

void testImageOpenAndSave()
{
	Mat src, dst;

	src = imread("Images/Lena_24bits.bmp", CV_LOAD_IMAGE_COLOR);	// Read the image

	if (!src.data)	// Check for invalid input
	{
		printf("Could not open or find the image\n");
		return;
	}

	// Get the image resolution
	Size src_size = Size(src.cols, src.rows);

	// Display window
	const char* WIN_SRC = "Src"; //window for the source image
	namedWindow(WIN_SRC, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Dst"; //window for the destination (processed) image
	namedWindow(WIN_DST, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_DST, src_size.width + 10, 0);

	cvtColor(src, dst, CV_BGR2GRAY); //converts the source image to a grayscale one

	imwrite("Images/Lena_24bits_gray.bmp", dst); //writes the destination to file

	imshow(WIN_SRC, src);
	imshow(WIN_DST, dst);

	printf("Press any key to continue ...\n");
	waitKey(0);
}

void testNegativeImage()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]
		
		Mat src = imread(fname,CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height,width,CV_8UC1);
		// Asa se acceseaaza pixelii individuali pt. o imagine cu 8 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i=0; i<height; i++)
		{
			for (int j=0; j<width; j++)
			{
				uchar val = src.at<uchar>(i,j);
				uchar neg = MAX_PATH-val;
				dst.at<uchar>(i,j) = neg;
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image",src);
		imshow("negative image",dst);
		waitKey();
	}
}

void testParcurgereSimplaDiblookStyle()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		int w = src.step; // no dword alignment is done !!!
		Mat dst = src.clone();

		double t = (double)getTickCount(); // Get the current time [s]

		// the fastest approach using the “diblook style”
		uchar *lpSrc = src.data;
		uchar *lpDst = dst.data;
		for (int i = 0; i<height; i++)
			for (int j = 0; j < width; j++) {
				uchar val = lpSrc[i*w + j];
				lpDst[i*w + j] = 255 - val;
				/* sau puteti scrie:
				uchar val = lpSrc[i*width + j];
				lpDst[i*width + j] = 255 - val;
				//	w = width pt. imagini cu 8 biti / pixel
				//	w = 3*width pt. imagini cu 24 biti / pixel
				*/
			}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image", src);
		imshow("negative image", dst);
		waitKey();
	}
}

void testColor2Gray()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src = imread(fname);

		int height = src.rows;
		int width = src.cols;

		Mat dst = Mat(height,width,CV_8UC1);
		
		// Asa se acceseaaza pixelii individuali pt. o imagine RGB 24 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i=0; i<height; i++)
		{
			for (int j=0; j<width; j++)
			{
				Vec3b v3 = src.at<Vec3b>(i,j);
				uchar b = v3[0];
				uchar g = v3[1];
				uchar r = v3[2];
				dst.at<uchar>(i,j) = (r+g+b)/3;
			}
		}
		
		imshow("input image",src);
		imshow("gray image",dst);
		waitKey();
	}
}

void testBGR2HSV()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);
		int height = src.rows;
		int width = src.cols;
		int w = src.step; // latimea in octeti a unei linii de imagine
		
		Mat dstH = Mat(height, width, CV_8UC1);
		Mat dstS = Mat(height, width, CV_8UC1);
		Mat dstV = Mat(height, width, CV_8UC1);
		
		// definire pointeri la matricele (8 biti/pixeli) folosite la afisarea componentelor individuale H,S,V
		uchar* dstDataPtrH = dstH.data;
		uchar* dstDataPtrS = dstS.data;
		uchar* dstDataPtrV = dstV.data;

		Mat hsvImg;
		cvtColor(src, hsvImg, CV_BGR2HSV);
		// definire pointer la matricea (24 biti/pixeli) a imaginii HSV
		uchar* hsvDataPtr = hsvImg.data;

		for (int i = 0; i<height; i++)
		{
			for (int j = 0; j<width; j++)
			{
				int hi = i*width * 3 + j * 3;
				// sau int hi = i*w + j * 3;	//w = 3*width pt. imagini 24 biti/pixel
				int gi = i*width + j;
				
				dstDataPtrH[gi] = hsvDataPtr[hi] * 510/360;		// H = 0 .. 255
				dstDataPtrS[gi] = hsvDataPtr[hi + 1];			// S = 0 .. 255
				dstDataPtrV[gi] = hsvDataPtr[hi + 2];			// V = 0 .. 255
			}
		}

		imshow("input image", src);
		imshow("H", dstH);
		imshow("S", dstS);
		imshow("V", dstV);
		waitKey();
	}
}

void testResize()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		Mat dst1,dst2;
		//without interpolation
		resizeImg(src,dst1,320,false);
		//with interpolation
		resizeImg(src,dst2,320,true);
		imshow("input image",src);
		imshow("resized image (without interpolation)",dst1);
		imshow("resized image (with interpolation)",dst2);
		waitKey();
	}
}

void testCanny()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src,dst,gauss;
		src = imread(fname,CV_LOAD_IMAGE_GRAYSCALE);
		int k = 0.4;
		int pH = 50;
		int pL = k*pH;
		GaussianBlur(src, gauss, Size(5, 5), 0.8, 0.8);
		Canny(gauss,dst,pL,pH,3);
		imshow("input image",src);
		imshow("canny",dst);
		waitKey();
	}
}

void testVideoSequence()
{
	VideoCapture cap("Videos/rubic.avi"); // off-line video from file
	//VideoCapture cap(0);	// live video from web cam
	if (!cap.isOpened()) {
		printf("Cannot open video capture device.\n");
		waitKey();
		return;
	}
		
	Mat edges;
	Mat frame;
	char c;

	while (cap.read(frame))
	{
		Mat grayFrame;
		cvtColor(frame, grayFrame, CV_BGR2GRAY);
		Canny(grayFrame,edges,40,100,3);
		imshow("source", frame);
		imshow("gray", grayFrame);
		imshow("edges", edges);
		c = cvWaitKey();  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished\n"); 
			break;  //ESC pressed
		};
	}
}


void testSnap()
{
	VideoCapture cap(0); // open the deafult camera (i.e. the built in web cam)
	if (!cap.isOpened()) // openenig the video device failed
	{
		printf("Cannot open video capture device.\n");
		return;
	}

	Mat frame;
	char numberStr[256];
	char fileName[256];
	
	// video resolution
	Size capS = Size((int)cap.get(CV_CAP_PROP_FRAME_WIDTH),
		(int)cap.get(CV_CAP_PROP_FRAME_HEIGHT));

	// Display window
	const char* WIN_SRC = "Src"; //window for the source frame
	namedWindow(WIN_SRC, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Snapped"; //window for showing the snapped frame
	namedWindow(WIN_DST, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_DST, capS.width + 10, 0);

	char c;
	int frameNum = -1;
	int frameCount = 0;

	for (;;)
	{
		cap >> frame; // get a new frame from camera
		if (frame.empty())
		{
			printf("End of the video file\n");
			break;
		}

		++frameNum;
		
		imshow(WIN_SRC, frame);

		c = cvWaitKey(10);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished");
			break;  //ESC pressed
		}
		if (c == 115){ //'s' pressed - snapp the image to a file
			frameCount++;
			fileName[0] = NULL;
			sprintf(numberStr, "%d", frameCount);
			strcat(fileName, "Images/A");
			strcat(fileName, numberStr);
			strcat(fileName, ".bmp");
			bool bSuccess = imwrite(fileName, frame);
			if (!bSuccess) 
			{
				printf("Error writing the snapped image\n");
			}
			else
				imshow(WIN_DST, frame);
		}
	}

}

int isInside(int a, int b)
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);

		int height = src.rows;
		int width = src.cols;
		printf("Lungime-Latime img: %d %d\n", height, width);

		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				if (a >= 0 && a <= height && b >= 0 && b <= width)	return 0;
				else return 1;
			}
		}
	}
	system("pause");
}

void MyCallBackFunc(int event, int x, int y, int flags, void* param)
{
	//More examples: http://opencvexamples.blogspot.com/2014/01/detect-mouse-clicks-and-moves-on-image.html
	Mat* src = (Mat*)param;

	if (event == CV_EVENT_LBUTTONDOWN)
		{
			printf("Pos(x,y): %d,%d  Color(RGB): %d,%d,%d\n",
				x, y,
				(int)(*src).at<Vec3b>(y, x)[2],
				(int)(*src).at<Vec3b>(y, x)[1],
				(int)(*src).at<Vec3b>(y, x)[0]);

		
		}
}

void MyCallBackFunc_L4(int event, int x, int y, int flags, void* param)
{
	Mat* src = (Mat*)param;
	if (event == CV_EVENT_LBUTTONDOWN) {
		//matricea de etichete
		Mat labels = Mat::zeros((*src).size(), CV_8UC1);
		std::queue<Point> que;

		int T = 14;
		int w = 3;

		double hue_avg = 0.0;
		int k = 1;//ethiceta curenta
		//adaug element start in coada
		que.push(Point(x, y));
		hue_avg = (*src).at<uchar>(y, x);
		// acesta primeste eticheta k
		labels.at<uchar>(y, x) = k;
		int N = 1;// numarul de pixeli din regiune
		//cat timp coada nu e goala
		while (!que.empty()) {
			// Retine poz. celui mai vechi element din coada
			Point oldest = que.front();
			// scoate element din coada
			que.pop();
			int xx = oldest.x;   // coordonatele lui
			int yy = oldest.y;
			// Pentru fiecare vecin al pixelului (xx, yy) ale carui coordonate
			for (int i = yy - w; i <= yy + w; i++) {
				for (int j = xx - w; j <= xx + w; j++) {
					// sunt in interiorul imaginii
					if (j > 0 && i > 0 && j < (*src).cols && i < (*src).rows) {
						// Daca abs(hue(vecin) – Hue_avg) < T si labels(vecin) == 0
						//double T = k * hue_avg;
						if ((abs((*src).at<uchar>(i, j) - hue_avg) < T) && (labels.at <uchar>(i, j) == 0)) {
							// Aduga vecin la regiunea curenta
							que.push(Point(j, i));
							// labels(vecin) = k
							labels.at <uchar>(i, j) = k;
							// Actualizeaza Hue_avg (medie ponderata)
							hue_avg = ((N)*hue_avg + abs((*src).at<uchar>(i, j))) / (N + 1);
							//incrementez N
							N++;
						}
					}
				}
			}
		}
		//imagine finala
		Mat dst = Mat::zeros((*src).size(), CV_8UC3);
		//parcurg imaginea 
		for (int i = 0; i < dst.rows; i++) {
			for (int j = 0; j < dst.cols; j++) {
				if (labels.at<uchar>(i, j) != 0) {
					dst.at<Vec3b>(i, j)[0] = 255;
					dst.at<Vec3b>(i, j)[1] = 255;
					dst.at<Vec3b>(i, j)[2] = 255;
				}
				else {
					dst.at<Vec3b>(i, j)[2] = 0;
					dst.at<Vec3b>(i, j)[1] = 0;
					dst.at<Vec3b>(i, j)[0] = 0;
				}
			}
		}
		imshow("Region growing", dst);


		Mat post = dst.clone();
		Mat element1 = getStructuringElement(MORPH_CROSS, Size(3, 3));
		erode(post, post, element1, Point(-1, -1), 2);
		dilate(post, post, element1, Point(-1, -1), 4);
		erode(post, post, element1, Point(-1, -1), 2);
		imshow("Procesare", post);
	}

}

void regionGrowing()
{
	Mat src;
	Mat hsv;
	Mat channels[3];
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname);
		int height = src.rows;
		int width = src.cols;
		GaussianBlur(src, src, Size(5, 5), 0.5, 0.5);
		Mat H = Mat(height, width, CV_8UC1);
		uchar* ph = H.data;
		cvtColor(src, hsv, CV_BGR2HSV);
		uchar* Hdata = hsv.data;
		split(hsv, channels);
		H = channels[0] * 255 / 180;
		imshow("src", src);
		setMouseCallback("src", MyCallBackFunc_L4, &H);

		waitKey(0);
	}

}

void testMouseClick()
{
	Mat src;
	// Read image from file 
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname);
		//Create a window
		namedWindow("My Window", 1);

		//set the callback function for any mouse event
		setMouseCallback("My Window", MyCallBackFunc, &src);

		//show the image
		imshow("My Window", src);

		// Wait until user press some key
		waitKey(0);
	}
}

void L3_ColorModel_Init()
{
	memset(histG_hue, 0, sizeof(unsigned int) * MAX_HUE);
}
Point Pstart, Pend; // Punctele/colturile aferente selectiei ROI curente 


void CallBackFuncL3(int event, int x, int y, int flags, void* userdata)
{
	Mat* H = (Mat*)userdata;
	Rect roi; // regiunea de interes curenta (ROI)
	if (event == EVENT_LBUTTONDOWN)
	{
		
		
			// punctul de start al ROI
			Pstart.x = x;
			Pstart.y = y;
			printf("Pstart: (%d, %d) ", Pstart.x, Pstart.y);
		

	}
	else if (event == EVENT_RBUTTONDOWN)
	{
		// punctul de final (diametral opus) al ROI
		Pend.x = x;
		Pend.y = y;
		printf("Pend: (%d, %d) ", Pend.x, Pend.y);
		// sortare puncte dupa x si y
		//(parametrii width si height ai structurii Rect > 0)
		roi.x = min(Pstart.x, Pend.x);
		roi.y = min(Pstart.y, Pend.y);
		roi.width = abs(Pstart.x - Pend.x);
		roi.height = abs(Pstart.y - Pend.y);
		printf("Local ROI: (%d, %d), (%d, %d)\n", roi.x, roi.y, roi.x + roi.width,
			roi.y + roi.height);
		int hist_hue[MAX_HUE]; // histograma locala a lui Hue
		memset(hist_hue, 0, MAX_HUE * sizeof(int));
		// Din toata imaginea H se selecteaza o subimagine (Hroi) aferenta ROI
		Mat Hroi = (*H)(roi);
		uchar hue;
		//construieste histograma locala aferente ROI
		for (int y = 0; y < roi.height; y++)
			for (int x = 0; x < roi.width; x++)
			{
				hue = Hroi.at<uchar>(y, x);
				hist_hue[hue]++;
			}
		//acumuleaza histograma locala in cea globala
		for (int i = 0; i < MAX_HUE; i++)
			histG_hue[i] += hist_hue[i];
		// afiseaza histohrama locala
		showHistogram("H local histogram", hist_hue, MAX_HUE, 200, true);
		// afiseaza histohrama globala / cumulativa
		showHistogram("H global histogram", histc_hue, MAX_HUE, 200, true);
	}
}


void L3_ColorModel_Build()
{
	Mat src;
	Mat hsv;
	// Read image from file
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname);
		int height = src.rows;
		int width = src.cols;
		// Aplicare FTJ gaussian pt. eliminare zgomote: essential sa il aplicati
		GaussianBlur(src, src, Size(5, 5), 0.8, 0.8);
		//Creare fereastra pt. afisare
		namedWindow("src", 1);
		// Componenta de culoare Hue a modelului HSV
		Mat H = Mat(height, width, CV_8UC1);
		// definire pointeri la matricea (8 biti/pixeli) folosita la stocarea
		// componentei individuale H
		uchar* lpH = H.data;
		cvtColor(src, hsv, CV_BGR2HSV); // conversie RGB -> HSV
		// definire pointer la matricea (24 biti/pixeli) a imaginii HSV
		uchar* hsvDataPtr = hsv.data;
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				// index in matricea hsv (24 biti/pixel)
				int hi = i * width * 3 + j * 3;
				int gi = i * width + j; // index in matricea H (8 biti/pixel)
				lpH[gi] = hsvDataPtr[hi] * 510 / 360; // lpH = 0 .. 255
			}
		}
		// Asociere functie de tratare a avenimentelor MOUSE cu ferestra curenta
		// Ultimul parametru este matricea H (valorile compunentei Hue)
		setMouseCallback("src", MyCallBackFunc, &H);
		imshow("src", src);
		// Wait until user press some key
		waitKey(0);
	}
}

void L3_ColorModel_Save()
{
	int hue, sat, i, j;
	int histF_hue[MAX_HUE]; // histograma filtrata cu FTJ
	memset(histF_hue, 0, MAX_HUE * sizeof(unsigned int));
#define FILTER_HISTOGRAM 1
#if FILTER_HISTOGRAM == 1
	// filtrare histograma cu filtru gaussian 1D de dimensiune w=7
	float gauss[7];
	float sqrt2pi = sqrtf(2 * PI);
	float sigma = 1.5;
	float e = 2.718;
	float sum = 0;
	// Construire gaussian
	for (i = 0; i < 7; i++) {
		gauss[i] = 1.0 / (sqrt2pi * sigma) * powf(e, -(float)(i - 3) * (i - 3)
			/ (2 * sigma * sigma));
		sum += gauss[i];
	}
	// Filtrare cu gaussian
	for (j = 3; j < MAX_HUE - 3; j++)
	{
		for (i = 0; i < 7; i++)
			histF_hue[j] += (float)histc_hue[j + i - 3] * gauss[i];
	}
#elif
	for (j = 0; j < MAX_HUE; j++)
		histF_hue[j] = histc_hue[j];
#endif // End of "Filtrare Gaussiana Histograma Hue"

	showHistogram("H global histogram", histc_hue, MAX_HUE, 200, true);
	showHistogram("H global filtered histogram", histF_hue, MAX_HUE, 200, true);
	// Wait until user press some key
	waitKey(0);
} //end of L3_ColorModel_Save()

void clasificare_pixeli()
{
	Mat src;
	Mat hsv;
	// Read image from file
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname);
		int height = src.rows;
		int width = src.cols;

		Mat dst = Mat(height, width, CV_8UC1);
		// Aplicare FTJ gaussian pt. eliminare zgomote: essential sa il aplicati
		GaussianBlur(src, src, Size(5, 5), 0, 0);
		//Creare fereastra pt. afisare
		namedWindow("src", 1);
		// Componenta de culoare Hue a modelului HSV
		Mat H = Mat(height, width, CV_8UC1);
		// definire pointeri la matricea (8 biti/pixeli) folosita la stocarea
		// componentei individuale H
		uchar* lpH = H.data;
		cvtColor(src, hsv, CV_BGR2HSV); // conversie RGB -> HSV
		// definire pointer la matricea (24 biti/pixeli) a imaginii HSV
		uchar* hsvDataPtr = hsv.data;
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				// index in matricea hsv (24 biti/pixel)
				int hi = i * width * 3 + j * 3;
				int gi = i * width + j; // index in matricea H (8 biti/pixel)
				lpH[gi] = hsvDataPtr[hi] * 510 / 360; // lpH = 0 .. 255
			}
		}

		int hue_mean = 16;
		int hue_std = 5;
		int k;
		printf("Dati un k=");
		scanf("%d", &k);

		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				if(0<= hue_mean - k * hue_std && hue_mean - k * hue_std <= MAX_HUE && 0<= hue_mean + k * hue_std && hue_mean + k * hue_std <= MAX_HUE)
					if (H.at<uchar>(i, j) >= hue_mean - k * hue_std && H.at<uchar>(i, j) <= hue_mean + k * hue_std)
						dst.at<uchar>(i, j) = 255;
					else dst.at<uchar>(i, j) = 0;
			}
		}
		imshow("Sursa", src);
		imshow("Clasificare", dst);

		// creare element structural de dimensiune 3x3 de tip patrat (V8)
		Mat element2 = getStructuringElement(MORPH_RECT, Size(3, 3));
		erode(dst, dst, element2, Point(-1, -1), 2);
		dilate(dst, dst, element2, Point(-1, -1), 4);
		erode(dst, dst, element2, Point(-1, -1), 2);
		imshow("Fara zgomot", dst);
	    imwrite("z.png", dst);
		//return dst;
		Labeling("Contur - functii din OpenCV", dst, false);
		
	
	
		waitKey(0);
	}
}

void lab4() 
{
	Mat src;
	Mat hsv;
	// Read image from file
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname);
		int height = src.rows;
		int width = src.cols;

		Mat dst = Mat(height, width, CV_8UC1);
		// Aplicare FTJ gaussian pt. eliminare zgomote: essential sa il aplicati
		GaussianBlur(src, src, Size(5, 5), 0, 0);
		//Creare fereastra pt. afisare
		imshow("zgomat gaussian", src);
		waitKey(0);
	}
}

void lab5() {	
	// Read image from file
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src_img = imread(fname, CV_LOAD_IMAGE_COLOR);
		Mat dst = src_img.clone();
		Mat src;
		cvtColor(src_img, src, CV_BGR2GRAY);
		GaussianBlur(src, src, Size(5, 5), 0, 0);

		int height = src.rows;
		int width = src.cols;
	
		// Lista/vector care va contine coordonatele (x,y) ale colturilor detectate (output)
		vector<Point2f> corners;
		// Nr. maxim de colturi luate in considerare. Daca nr. de colturi > maxCorners se vor
		//considera cele cu raspuns R maxim
		int maxCorners = 100;
		// Factor cu care se multiplica masura de calitate a celui mai bun colt (val. proprie
		//minima) pt. metoda Shi-Tomasi respectiv valoarea functiei de raspuns R (Harris)
		//ex: qualityMeasure = 1500, qualityLevel = 0.01 ⇒ colturile cu valoarea mai mica de
		//1500*0.01 sunt rejectate:
		double qualityLevel = 0.01;
		// Dimensiunea ferestrei w in care se calculeaza matricea de autocorelatie (covarianta a
		//derivatelor)
		// Distana euclidiana minima dintre 2 colturi returnate (functia elimina orice colt vecin cu
		//coltul curent aflat la o dstanta mai mica de 10 pixeli si care are masura de calitate mai
		//mica ∼ metoda Non - Maxima Supression)
		double minDistance = 10;
		int blockSize = 3; // 2,3, ...
		// Selectia metodei de detectie: Harris (true) sau Shi-Tomasi (false).
		bool useHarrisDetector = true;
		// Factorul k (vezi documentatia curs)
		double k = 0.04;
		// Apel functie
		goodFeaturesToTrack(src,
			corners,
			maxCorners,
			qualityLevel,
			minDistance,
			Mat(), //masca pt. ROI - optional
			blockSize,
			useHarrisDetector,
			k);
			
		int r=3; // raza cercului
		for (int i = 0; i < corners.size(); i++) {
				circle(dst, corners[i], r, Scalar(0, 255, 0), CV_FILLED, 1);
		}
		imshow("dst", dst);

		/// adaptare functie pt punctul 1 - coordonate rafinate ale colturilor

		//parametrii
		Size winSize = Size(5, 5);
		Size zeroZone = Size(-1, -1);

		TermCriteria criteria = TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 40, 0.001);
		cornerSubPix(src, corners, winSize, zeroZone, criteria);

		// scriu in fisier
		FILE* file = fopen("corners.txt", "w");
		for (int i = 0; i < corners.size(); i++)
		{
			fprintf(file, "%.2f,  %.2f \n", corners[i].x, corners[i].y);
		}
		fclose(file);

	}
}

void cornerHarris_demo(){
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, CV_LOAD_IMAGE_COLOR);
		Mat src_gray;
		cvtColor(src, src_gray, CV_BGRA2GRAY);

		int blockSize = 2;
		int apertureSize = 3;
		double k = 0.04;

		Mat dst = Mat::zeros(src.size(), CV_32FC1);		
		cornerHarris(src_gray, dst, blockSize, apertureSize, k, BORDER_DEFAULT);

		Mat dst_norm, dst_norm_scaled;
		normalize(dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
		convertScaleAbs(dst_norm, dst_norm_scaled);

		Mat dst_non_max = dst_norm_scaled.clone();

		int thresh = 150;
		for (int j = 0; j < dst_norm.rows; j++){
			for (int i = 0; i < dst_norm.cols; i++){
				if ((int)dst_norm.at<float>(j, i) > thresh){
					bool flag = true;
					for (int jj = -5; jj < 5; jj++) {
						for (int ii = -5; ii < 5; ii++){
							if (i + ii > 0 && i + ii < dst_norm.cols && j + jj > 0 && j + jj < dst_norm.rows) {
								if (dst_norm.at<float>(j, i) < dst_norm.at<float>(j + jj, i + ii))
									flag = false;
							}
						}
					}
					if (flag){
						circle(dst_norm_scaled, Point(i, j), 6, Scalar(255), 1, 8, 0);
					}
					circle(dst_non_max, Point(i, j), 6, Scalar(255), 1, 8, 0);
				}
			}
		}
		resize(dst_norm_scaled, dst_norm_scaled, Size(600, 600), INTER_LINEAR);
		imshow("cu  supresia non maximelor", dst_norm_scaled);
		resize(dst_non_max, dst_non_max, Size(600, 600), INTER_LINEAR);
		imshow("fara supresia non maximelor", dst_non_max);
		waitKey(0);
	}

}
void testVideoCorners()
{
	VideoCapture cap("D:/an4/IOC/Img_corners/rubic.avi"); 	
	if (!cap.isOpened()){
		printf("Cannot open video capture device.\n");
		waitKey(0);
		return;
	}

	Mat edges;
	Mat frame;
	char c;

	while (cap.read(frame)){
		Mat grayFrame;
		Mat dst = frame.clone();
		cvtColor(frame, grayFrame, CV_BGR2GRAY);
		vector<Point2f> corners;
		int maxCorners = 100;
		double qualityLevel = 0.01;
		double minDistance = 10;
		int blockSize = 3;
		bool useHarrisDetector = true;
		double k = 0.04;
		GaussianBlur(grayFrame, grayFrame, Size(5, 5), 0.5, 0.5);
		goodFeaturesToTrack(grayFrame,
			corners,
			maxCorners,
			qualityLevel,
			minDistance,
			Mat(), //masca pt. ROI - optional
			blockSize,
			useHarrisDetector,
			k);
		for (int i = 0; i < corners.size(); i++){
			DrawCross(dst, corners[i], 10, Scalar(0, 255, 0), 1);
		}
		imshow("dst", dst);
		c = cvWaitKey(0);  // waits a key press to advance to the next frame
		if (c == 27){
			// press ESC to exit
			printf("ESC pressed - capture finished\n");
			break;  //ESC pressed
		};
	}
}

void testVideo(int method)
{
	VideoCapture cap("Videos/Laboratory.avi");
	if (!cap.isOpened()) {
		printf("Cannot open video capture device.\n");
		waitKey(0);
		return;
	}
	Mat frame, gray;
	Mat backgnd;
	Mat diff;
	Mat dst;
	char c;
	int frameNum = -1;
	const unsigned char Th = 25;
	const double alpha = 0.05;

	for (;;) {
		cap >> frame; // achizitie frame nou
		double t = (double)getTickCount();
		if (frame.empty())
		{
			printf("End of video file\n");
			break;
		}
		++frameNum;
		if (frameNum == 0)
			imshow("sursa", frame);  // daca este primul cadru se afiseaza doar sursa

		cvtColor(frame, gray, CV_BGR2GRAY);
		//Optional puteti aplica si un FTJ Gaussian
		GaussianBlur(gray, gray, Size(5, 5), 0, 0);
		//Se initializeaza matricea / imaginea destinatie pentru fiecare frame
		dst = Mat::zeros(gray.size(), gray.type());
		const int channels_gray = gray.channels();
		//restrictionam utilizarea metodei doar pt. imagini grayscale cu un canal (8 bit / pixel)
		if (channels_gray > 1)
			return;

		if (frameNum > 0)  // daca nu este primul cadru
		{
			absdiff(gray, backgnd, diff);
			//------ SABLON DE PRELUCRARI PT. METODELE BACKGROUND SUBTRACTION -------
			// Calcul imagine diferenta dintre cadrul current (gray) si fundal (backgnd)
			// Rezultatul se pune in matricea/imaginea diff
			// Se actualizeaza matricea/imaginea model a fundalului (backgnd)
			// conform celor 3 metode:
			
			switch (method) {
			case 1: backgnd = gray.clone();
				for (int i = 0; i < diff.rows; i++) {
					for (int j = 0; j < diff.cols; j++) {
						if (diff.at<uchar>(i, j) > Th)
							dst.at<uchar>(i, j) = 255;
					}
				}
				break;
			case 2:  addWeighted(gray, alpha, backgnd, 1.0 - alpha, 0, backgnd);
				for (int i = 0; i < diff.rows; i++) {
					for (int j = 0; j < diff.cols; j++) {
						if (diff.at<uchar>(i, j) > Th)
							dst.at<uchar>(i, j) = 255;
					}
				}
				break;
			case 3:
				for (int i = 0; i < diff.rows; i++) {
					for (int j = 0; j < diff.cols; j++) {
						if (diff.at<uchar>(i, j) > Th)
							dst.at<uchar>(i, j) = 255;
						else
							backgnd.at<uchar>(i, j) =
							alpha * gray.at<uchar>(i, j) + (1.0 - alpha) * backgnd.at<uchar>(i, j);
					}
				}
				break;
			}
			imshow("source", frame);
			imshow("grayscale", gray);
			imshow("dst", dst);
			Mat element = getStructuringElement(MORPH_CROSS, Size(3, 3));
			erode(dst, dst, element, Point(-1, -1), 2);
			dilate(dst, dst, element, Point(-1, -1), 2);
			imshow("dst erosziune+ dilatare", dst);
			imshow("diff", diff);
		}
		else
			backgnd = gray.clone();
		t = ((double)getTickCount() - t) / getTickFrequency();
		printf("%d - %.3f [ms]\n", frameNum, t * 1000);
		c = cvWaitKey(0);
		if (c == 27) {
			printf("ESC pressed - playback finished\n");
			break;
		}
	}
}


void calcOpticalFlowHS(const Mat& prev, const Mat& crnt, float lambda, int n0, Mat& flow)
{
	Mat vx = Mat::zeros(crnt.size(), CV_32FC1);// matricea comp. x a fluxului optic
	Mat vy = Mat::zeros(crnt.size(), CV_32FC1); // matricea comp. y a fluxului optic
	Mat Et = Mat::zeros(crnt.size(), CV_32FC1);// derivatele temporale
	Mat Ex, Ey; // Matricele derivatelor spatiale (gradient)

	// Calcul componenta orizontala a gradientului
	Sobel(crnt, Ex, CV_32F, 1, 0);
	// Calcul componenta verticala a gradientului
	Sobel(crnt, Ey, CV_32F, 0, 1);
	// Calcul derivata temporala
	Mat prev_float, crnt_float; // matricile imaginii crnt sip rev se convertesc in float
	prev.convertTo(prev_float, CV_32FC1);
	crnt.convertTo(crnt_float, CV_32FC1);
	Et = crnt_float - prev_float;

	// Insercati codul aferent algoritmului Horn-Schunk
	for (int itNo = 0; itNo < n0; itNo++)
	{
		for (int i = 1; i < vx.rows - 1; i++)
		{
			for (int j = 1; j < vx.cols - 1; j++)
			{
				vx.at<float>(i, j) = (vx.at<float>(i - 1, j) + vx.at<float>(i, j + 1) +
					vx.at<float>(i, j - 1) + vx.at<float>(i + 1, j)) / 4;  //6.3
				vy.at<float>(i, j) = (vy.at<float>(i - 1, j) + vy.at<float>(i, j + 1) +
					vy.at<float>(i, j - 1) + vy.at<float>(i + 1, j)) / 4;  //6.4
				float alfa;
				alfa = lambda * (Ex.at<float>(i, j) * vx.at<float>(i, j) + Ey.at<float>(i, j) * vy.at<float>(i, j) + Et.at<float>(i, j))
					/ (1 + lambda * (Ex.at<float>(i, j) * Ex.at<float>(i, j) + Ey.at<float>(i, j) * Ey.at<float>(i, j)));  //6.5
				vx.at<float>(i, j) = vx.at<float>(i, j) - alfa * Ex.at<float>(i, j);  //6.6
				vy.at<float>(i, j) = vy.at<float>(i, j) - alfa * Ey.at<float>(i, j); //6.7
			}
		}

	}
	// Compune comp. x si y ale fluxului optic intr-o matrice cu elemente de tip Point2f
	flow = convert2flow(vx, vy);
	// Vizualizare rezultate intermediare:
// gradient,derivata temporala si componentele vectorilor de miscare sub forma unor
// imagini grayscale obtinute din matricile de tip float prin normalizare
	Mat Ex_gray, Ey_gray, Et_gray, vx_gray, vy_gray;
	normalize(Ex, Ex_gray, 0, 255, NORM_MINMAX, CV_8UC1, Mat());
	normalize(Ey, Ey_gray, 0, 255, NORM_MINMAX, CV_8UC1, Mat());
	normalize(Et, Et_gray, 0, 255, NORM_MINMAX, CV_8UC1, Mat());
	normalize(vx, vx_gray, 0, 255, NORM_MINMAX, CV_8UC1, Mat());
	normalize(vy, vy_gray, 0, 255, NORM_MINMAX, CV_8UC1, Mat());

	imshow("Ex", Ex_gray);
	imshow("Ey", Ey_gray);
	imshow("Et", Et_gray);
	imshow("vx", vx_gray);
	imshow("vy", vy_gray);
}

void Horn_Schunk()
{
	Mat frame;
	Mat crnt;
	Mat prev;
	Mat dst;
	Mat flow;
	char folderName[MAX_PATH];
	char fname[MAX_PATH];
	if (openFolderDlg(folderName) == 0)
		return;
	FileGetter fg(folderName, "bmp");
	
	int frameNum = -1; //current frame counter
	int n = 8;
	float lambda = 10.0f;
	int c;

	while (fg.getNextAbsFile(fname))// citeste in fname numele caii complete
									// la cate un fisier bitmap din secventa
	{
		crnt = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		cvMoveWindow("DST", 10 + crnt.cols, 0);
		GaussianBlur(crnt, crnt, Size(5, 5), 0.8, 0.8);
		++frameNum;


		if (frameNum > 0)  // not the first frame
		{
			// functii de procesare (calcul flux optic) si afisare
			// Horn-Shunk
			double t = (double)getTickCount();
			//calcOpticalFlowHS(prev, crnt, 0, 0.1, TermCriteria(TermCriteria::MAX_ITER, 16, 0), flow);
			calcOpticalFlowHS(prev, crnt, lambda, n, flow);
			// Stop the proccesing time measure
			t = ((double)getTickCount() - t) / getTickFrequency();
			printf("%d - %.3f [ms]\n", frameNum, t * 1000);
			showFlow("Dst", prev, flow, 1, 1.5, true, true, false);

		}
		imshow("Src", crnt);
		// store crntent frame as previos for the next cycle
		prev = crnt.clone();
		c = cvWaitKey(0);// press any key to advance between frames
		//for continous play use cvWaitKey( delay > 0)
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - playback finished\n\n");
			break; //ESC pressed
		}
	}
}


void calcOpticalFlowPyrLK() {

	VideoCapture cap("Videos/S3.avi");
	if (!cap.isOpened()) {
		printf("Cannot open video capture device.\n");
		waitKey();
		return;
	}

	Mat frame, crnt;
	Mat prev;
	Mat dst;
	Mat flow;
	char c;

	int frameNum = -1;

	for (;;) {
		cap >> frame;
		if (frame.empty())
		{
			printf("End of video file\n");
			break;
		}
		++frameNum;
		if (frameNum == 0)
			imshow("sursa", frame);
		cvtColor(frame, crnt, CV_BGR2GRAY);

		GaussianBlur(crnt, crnt, Size(5, 5), 0.8, 0.8);

		if (frameNum > 0)
		{
			// parameters for calcOpticalFlowPyrLK
			vector<Point2f> prev_pts; // vector of 2D points with previous image features
			vector<Point2f> crnt_pts;// vector of 2D points with current image (matched) features
			vector<uchar> status; // output status vector: 1 if the wlow for the corresponding
								//feature was found. 0 otherwise
			vector<float> error; // output vector of errors; each element of the vector is set to
								//an error for the corresponding feature
			Size winSize = Size(21, 21); // size of the search window at each pyramid level - deafult
										//(21, 21)
			int maxLevel = 3; // maximal pyramid level number - deafult 3
				//parameter, specifying the termination criteria of the iterative search algorithm
				// (after the specified maximum number of iterations criteria.maxCount or when the search
				//window moves by less than criteria.epsilon
				// deafult 30, 0.01
			TermCriteria criteria = TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 20, 0.03);
			int flags = 0;
			double minEigThreshold = 1e-4;

			int maxCorners = 100;
			double qualityLevel = 0.01;
			double minDistance = 10;
			int blockSize = 3;
			bool useHarrisDetector = true;
			double k = 0.04;
			vector<Point2f> corners;			

			// Apply corner detection
			goodFeaturesToTrack(prev,
				prev_pts,
				maxCorners,
				qualityLevel,
				minDistance,
				Mat(),
				blockSize,
				useHarrisDetector,
				k);
			calcOpticalFlowPyrLK(prev, crnt, prev_pts, crnt_pts, status, error, winSize, maxLevel, criteria);
			showFlowSparse("Dst", prev, prev_pts, crnt_pts, status, error, 2, true, true, true);
		}
		prev = crnt.clone();
		c = cvWaitKey(0);
		if (c == 27) {
			printf("ESC pressed - playback finished\n\n");
			break;
		}
	}
}
/* Histogram display function - display a histogram using bars (simlilar tu L3 / PI)
Input:
name - destination (output) window name
hist - pointer to the vector containing the histogram values
hist_cols - no. of bins (elements) in the histogram = histogram image width
hist_height - height of the histogram image
Call example:
showHistogram (WIN_HIST, hist_dir, 360, 200, true);
*//*
void showHistogram(const string& name, int* hist, const int hist_cols, const int
	hist_height, bool showImages = true)
{
	if (showImages)
	{
		Mat imgHist(hist_height, hist_cols, CV_8UC3, CV_RGB(255, 255, 255));
		//computes histogram maximum
		int max_hist = 0;
		for (int i = 0; i < hist_cols; i++)
			if (hist[i] > max_hist)
				max_hist = hist[i];
		double scale = 1.0;
		scale = (double)hist_height / max_hist;
		int baseline = hist_height - 1;
		for (int x = 0; x < hist_cols; x++) {
			Point p1 = Point(x, baseline);
			Point p2 = Point(x, baseline - cvRound(hist[x] * scale));
			line(imgHist, p1, p2, CV_RGB(255, 0, 255));
		}

		imshow(name, imgHist);
	}
}*/
/* Optical flow directions histogram display function - display a histogram using bars
colored in Middlebury color coding
Input:
name - destination (output) window name
hist - pointer to the vector containing the histogram values
hist_cols - no. of bins (elements) in the histogram = histogram image width
7
hist_height - height of the histogram image
Call example:
showHistogramDir (WIN_HIST, hist_dir, 360, 200, true);
*//*
void showHistogramDir(const string& name, int* hist, const int hist_cols, const int
	hist_height, bool showImages = true)
{
	unsigned char r, g, b;
	if (showImages)
	{
		Mat imgHist(hist_height, hist_cols, CV_8UC3, CV_RGB(255, 255, 255));
		//computes histogram maximum
		int max_hist = 0;
		for (int i = 0; i < hist_cols; i++)
			if (hist[i] > max_hist)
				max_hist = hist[i];
		double scale = 1.0;
		scale = (double)hist_height / max_hist;
		int baseline = hist_height - 1;
		for (int x = 0; x < hist_cols; x++) {
			Point p1 = Point(x, baseline);
			Point p2 = Point(x, baseline - cvRound(hist[x] * scale));
			r = HSI2RGB[x][R];
			g = HSI2RGB[x][G];
			b = HSI2RGB[x][B];
			line(imgHist, p1, p2, CV_RGB(r, g, b));
		}

		imshow(name, imgHist);
	}
}*/
void flux_optic() {

	Mat crnt; // current frame red as grayscale (crnt)
	Mat prev; // previous frame (grayscale)
	Mat dst; // output image/frame
	Mat flow; // flow - matrix containing the optical flow vectors/pixel

	char folderName[MAX_PATH];
	char fname[MAX_PATH];
	if (openFolderDlg(folderName) == 0)
		return;
	FileGetter fg(folderName, "bmp");

	int frameNum = -1;//current frame counter

	makeColorwheel();
	make_HSI2RGB_LUT();

	while (fg.getNextAbsFile(fname))// citeste in fname numele caii complete
									// la cate un fisier bitmap din secventa
	{
		crnt = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		double t = (double)getTickCount();// Get the crntent time [s]
		GaussianBlur(crnt, crnt, Size(5, 5), 0.8, 0.8);
		++frameNum;

		if (frameNum > 0)// not the first frame
		{
			// functii de procesare (calcul flux optic) si afisare
			int winSize = 11;
			calcOpticalFlowFarneback(prev, crnt, flow, 0.5, 3, winSize, 10, 6, 1.5, 0);
			showFlowDense("Img", crnt, flow, 1, 1);

			//histograma vector intregi reinitializata la 0
			int hist_dir[360] = { 0 };
			
			//pt fiecare pixel calculez unghiul facut de de vector cu axa Ox
			for (int i = 0; i < crnt.rows; i++) {
				for (int j = 0; j < crnt.cols; j++) {
					Point2f f = flow.at<Point2f>(i, j); // vectorul de miscare in punctual (r,c)
					// vectorul de miscare al punctului se considera cu originea in imaginea trecuta (prev)
					// si varful in imaginea curenta (crnt) –> se iau valorile lui din vectorul flow cu minus !
					float dir_rad = CV_PI + atan2(-f.y, -f.x); //directia vectorului in radiani
					int dir_deg = (dir_rad * 180) / CV_PI;

					//calculez modului lui v
					float modul = sqrt(pow(f.x, 2) + pow(f.y, 2));
					//filtare pe baza pragului minVel
					if (dir_deg >= 0 && dir_deg < 360) {
						if (modul >= 1) {
							hist_dir[dir_deg]++;
						}
					}
				}
			}
			showHistogramDir("Hist", hist_dir, 360, 200, true);
			// 200 [pixeli] = inaltimea ferestrei de afisare a histogramei

		}
		prev = crnt.clone();
		// Get the crntent time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("%d - %.3f [ms]\n", frameNum, t * 1000);
		prev = crnt.clone();
		int c = cvWaitKey(0);
		if (c == 27) {
			printf("ESC pressed - playback finished\n\n");
		}
	}
}
/* ------------------------------------------------------------------------------------
---
Detects all the faces and eyes in the input image
window_name - name of the destination window in which the detection results are
displayed
frame - source image
minFaceSize - minimum size of the ROI in which a Face is searched
minEyeSize - minimum size of the ROI in which an Eye is searched
acording to the antropomorphic features of a face, minEyeSize = minFaceSize / 5
Usage: FaceDetectandDisplay( “Dst”, dst, minFaceSize, minEyeSize );
---------------------------------------------------------------------------------------
- */
#include "opencv2/objdetect/objdetect.hpp"
CascadeClassifier face_cascade; // cascade clasifier object for face
CascadeClassifier eyes_cascade; // cascade clasifier object for eyes
CascadeClassifier nose_cascade;
CascadeClassifier mouth_cascade;
CascadeClassifier mcs_upperbody_cascade;
CascadeClassifier fullbody_cascade;
CascadeClassifier lowerbody_cascade;
CascadeClassifier upperbody_cascade;
void FaceDetectandDisplay(const string& window_name, Mat frame,int minFaceSize, int minEyeSize)
{
	std::vector<Rect> faces;
	Mat frame_gray;
	cvtColor(frame, frame_gray, CV_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);
	//-- Detect faces
	face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE,
		Size(minFaceSize, minFaceSize));
	for (int i = 0; i < faces.size(); i++) {
		// get the center of the face
		Point center(faces[i].x + faces[i].width * 0.5, faces[i].y + faces[i].height * 0.5);
		// draw circle around the face
		ellipse(frame, center, Size(faces[i].width * 0.5, faces[i].height * 0.5), 0, 0,360, Scalar(255, 0, 255), 4, 8, 0);
		Mat faceROI = frame_gray(faces[i]);
		std::vector<Rect> eyes;
		//-- In each face (rectangular ROI), detect the eyes
		eyes_cascade.detectMultiScale(faceROI, eyes, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE,
			Size(minEyeSize, minEyeSize));
		for (int j = 0; j < eyes.size(); j++)
		{
			// get the center of the eye
			
			Point center(faces[i].x + eyes[j].x + eyes[j].width * 0.5,
				faces[i].y + eyes[j].y + eyes[j].height * 0.5);
			int radius = cvRound((eyes[j].width + eyes[j].height) * 0.25);
			// draw circle around the eye
			circle(frame, center, radius, Scalar(255, 0, 0), 4, 8, 0);

		}
	}
	imshow(window_name, frame);  //-- Show what you got
	waitKey();
}
void face_detection_eyes() {
	String face_cascade_name = "haarcascade_frontalface_alt.xml";
	String eyes_cascade_name = "haarcascade_eye_tree_eyeglasses.xml";

	// Load the cascades
	if (!face_cascade.load(face_cascade_name))
	{
		printf("Error loading face cascades !\n");
		return;
	}

	if (!eyes_cascade.load(eyes_cascade_name))
	{
		printf("Error loading eyes cascades !\n");
		return;
	}


	Mat src;
	Mat dst;
	char fname[MAX_PATH];

	while (openFileDlg(fname))
	{

		src = imread(fname, CV_LOAD_IMAGE_COLOR);
		dst = src.clone();
		int minFaceSize = 30;
		int minEyeSize = minFaceSize / 5; // conform proprietatilor antropomorfice ale fetei(idem pt.gura si nas)
		FaceDetectandDisplay("Dst", dst, minFaceSize, minEyeSize);
	}
}

void FaceDetectandDisplayMouthNose(const string& window_name, Mat frame,
	int minFaceSize, int minEyeSize)
{
	std::vector<Rect> faces;
	Mat frame_gray;
	cvtColor(frame, frame_gray, CV_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);
	face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE,
		Size(minFaceSize, minFaceSize));
	for (int i = 0; i < faces.size(); i++)
	{
		Point center(faces[i].x + faces[i].width * 0.5, faces[i].y + faces[i].height * 0.5);
		ellipse(frame, center, Size(faces[i].width * 0.5, faces[i].height * 0.5), 0, 0,
		360, Scalar(0, 255, 255), 4, 8, 0);
		//rectangle(frame, faces[i], Scalar(0, 255, 255), 4, 8, 0);
		Mat faceROI = frame_gray(faces[i]);
		std::vector<Rect> eyes;
		eyes_cascade.detectMultiScale(faceROI, eyes, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE,
			Size(minEyeSize, minEyeSize));
		for (int j = 0; j < eyes.size(); j++)

		{
			Point center(faces[i].x + eyes[j].x + eyes[j].width * 0.5,
				faces[i].y + eyes[j].y + eyes[j].height * 0.5);
			int radius = cvRound((eyes[j].width + eyes[j].height) * 0.25);
			//	rectangle(frame, eyes[j], Scalar(0, 255, 0), 4, 8, 0);
			circle(frame, center, radius, Scalar(0, 255, 0), 4, 8, 0);
		}


		Rect nose_rect; //nose is the 40% ... 75% height of the face
		nose_rect.x = faces[i].x;
		nose_rect.y = faces[i].y + 0.4 * faces[i].height;
		nose_rect.width = faces[i].width;
		nose_rect.height = 0.35 * faces[i].height;

		Mat nose_ROI = frame_gray(nose_rect);
		std::vector<Rect> nose;

		nose_cascade.detectMultiScale(nose_ROI, nose, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE,
			Size(minFaceSize / 5, minFaceSize / 5));

		for (int j = 0; j < nose.size(); j++) {
			Point center(nose_rect.x + nose[j].x + nose[j].width * 0.5,
				nose_rect.y + nose[j].y + nose[j].height * 0.5);

			int radius = cvRound((nose[j].width + nose[j].height) * 0.25);
			circle(frame, center, radius, Scalar(0, 0, 255), 4, 8, 0);
		}


		Rect mouth_rect; //mouth is in the 70% ... 99% height of the face
		mouth_rect.x = faces[i].x;
		mouth_rect.y = faces[i].y + 0.7 * faces[i].height;
		mouth_rect.width = faces[i].width;
		mouth_rect.height = 0.29 * faces[i].height;
		std::vector<Rect> mouth;
		Mat mouth_ROI = frame_gray(mouth_rect);

		mouth_cascade.detectMultiScale(mouth_ROI, mouth, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE,
			Size(minFaceSize / 4, minFaceSize / 4));

		for (int j = 0; j < mouth.size(); j++) {
			Point center(mouth_rect.x + mouth[j].x + mouth[j].width * 0.5,
				mouth_rect.y + mouth[j].y + mouth[j].height * 0.5);

			int radius = cvRound((mouth[j].width + mouth[j].height) * 0.25);

			circle(frame, center, radius, Scalar(255, 51, 153), 4, 8, 0);
		}
	}
	imshow(window_name, frame);
	waitKey();

}


void face_detection_mouth_nose() {
	String face_cascade_name = "haarcascade_frontalface_alt.xml";
	String eyes_cascade_name = "haarcascade_eye_tree_eyeglasses.xml";
	String mouth_cascade_name = "haarcascade_mcs_mouth.xml";
	String nose_cascade_name = "haarcascade_mcs_nose.xml";

	if (!face_cascade.load(face_cascade_name))
	{
		printf("Error loading face cascades !\n");
		return;
	}

	if (!eyes_cascade.load(eyes_cascade_name))
	{
		printf("Error loading eyes cascades !\n");
		return;
	}

	if (!mouth_cascade.load(mouth_cascade_name))
	{
		printf("Error loading mouth cascades !\n");
		return;
	}

	if (!nose_cascade.load(nose_cascade_name))
	{
		printf("Error loading nose cascades !\n");
		return;
	}

	Mat src;
	Mat dst;
	char fname[MAX_PATH];

	while (openFileDlg(fname))
	{
		src = imread(fname, CV_LOAD_IMAGE_COLOR);
		dst = src.clone();
		int minFaceSize = 30;
		int minEyeSize = minFaceSize / 5;
		FaceDetectandDisplayMouthNose("Dst", dst, minFaceSize, minEyeSize);
	}
}


void face_detect_video() {
	String face_cascade_name = "haarcascade_frontalface_alt.xml";
	String eyes_cascade_name = "haarcascade_eye_tree_eyeglasses.xml";

	int minFaceSize = 30;
	int minEyeSize = minFaceSize / 5;
	if (!face_cascade.load(face_cascade_name))
	{
		printf("Error loading face cascades !\n");
		return;
	}

	if (!eyes_cascade.load(eyes_cascade_name))
	{
		printf("Error loading eyes cascades !\n");
		return;
	}

	char c;
	VideoCapture cap("D:/an4/IOC/Face/Megamind.avi");
	if (!cap.isOpened()) {
		printf("Cannot open video capture device.\n");
		waitKey();
		return;
	}
	Mat frame;
	for (;;) {
		double t = (double)getTickCount();
		cap >> frame;
		if (frame.empty())
		{
			printf("End of video file\n");
			break;
		}
		t = ((double)getTickCount() - t) / getTickFrequency();
		printf("Curent frame time: %.3f [ms]\n", t * 1000);
		FaceDetectandDisplay("Dst", frame, minFaceSize, minEyeSize);

		c = cvWaitKey();
		if (c == 27) {
			printf("ESC pressed - capture finished\n");
			break;
		};
	}
}


void FaceDetectandDisplay2(const string& window_name, Mat frame,
	int minFaceSize, int minEyeSize)
{
	std::vector<Rect> faces;
	Mat frame_gray;
	cvtColor(frame, frame_gray, CV_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);
	face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE,
		Size(minFaceSize, minFaceSize));
	for (int i = 0; i < faces.size(); i++)
	{
		//rectangle(frame, faces[i], Scalar(0, 255, 255), 4, 8, 0);
		Point center(faces[i].x + faces[i].width * 0.5, faces[i].y + faces[i].height * 0.5);
		ellipse(frame, center, Size(faces[i].width * 0.5, faces[i].height * 0.5), 0, 0,
			360, Scalar(255, 0, 255), 4, 8, 0);
	}
	imshow(window_name, frame);
	waitKey();
}


void face_detection_lbp() {
	String face_cascade_name = "lbpcascade_frontalface.xml";

	if (!face_cascade.load(face_cascade_name))
	{
		printf("Error loading face cascades !\n");
		return;
	}

	Mat src;
	Mat dst;
	char fname[MAX_PATH];


	while (openFileDlg(fname))
	{

		src = imread(fname, CV_LOAD_IMAGE_COLOR);
		dst = src.clone();
		int minFaceSize = 30;
		int minEyeSize = minFaceSize / 5;
		FaceDetectandDisplay2("Dst", dst, minFaceSize, minEyeSize);
	}
}

void FullBodyDetectandDisplay(const string& window_name, Mat frame, int minFaceSize, int minEyeSize)
{
	std::vector<Rect> faces;
	Mat frame_gray;
	cvtColor(frame, frame_gray, CV_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);
	//-- Detect faces
	int minBodyHight = 100;
	upperbody_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE,
		Size(0.3,minBodyHight));
	for (int i = 0; i < faces.size(); i++) {
		// get the center of the face
		Point center(faces[i].x + faces[i].width * 0.5, faces[i].y + faces[i].height * 0.5);
		// draw circle around the face
		ellipse(frame, center, Size(faces[i].width * 0.5, faces[i].height * 0.5), 0, 0, 360, Scalar(255, 0, 255), 4, 8, 0);
		Mat faceROI = frame_gray(faces[i]);
		std::vector<Rect> eyes;
		//-- In each face (rectangular ROI), detect the eyes
		fullbody_cascade.detectMultiScale(faceROI, eyes, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE,
			Size(minEyeSize, minEyeSize));
		for (int j = 0; j < eyes.size(); j++)
		{
			// get the center of the eye

			Point center(faces[i].x + eyes[j].x + eyes[j].width * 0.5,
				faces[i].y + eyes[j].y + eyes[j].height * 0.5);
			int radius = cvRound((eyes[j].width + eyes[j].height) * 0.25);
			// draw circle around the eye
			circle(frame, center, radius, Scalar(255, 0, 0), 4, 8, 0);

		}
	}
	imshow(window_name, frame);  //-- Show what you got
	waitKey();
}

void Fulldetection(const string& window_name, Mat frame,
	int minFaceSize, int minEyeSize)
{
	//full body
	std::vector<Rect> bodies;
	int minBodyHeight = 120;
	Mat frame_gray;
	Rect fullbody;
	cvtColor(frame, frame_gray, CV_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);
	fullbody_cascade.detectMultiScale(frame_gray, bodies, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE,
		Size(minBodyHeight * 0.3f, minBodyHeight));
	for (int i = 0; i < bodies.size(); i++)
	{
		//Point center(bodies[i].x + bodies[i].width/2 , bodies[i].y + bodies[i].height /2);
		//ellipse(frame, center, Size(bodies[i].width * 0.5, bodies[i].height * 0.5), 0, 0,
			//360, Scalar(0, 255, 255), 4, 8, 0);
		Point center(bodies[i].x + bodies[i].width / 2, bodies[i].y + bodies[i].height / 2);
		Point first(center.x - (bodies[i].width / 2), center.y - (bodies[i].height / 2));
		Point second(center.x + (bodies[i].width / 2), center.y + (bodies[i].height / 2));
		fullbody = Rect(first, second);
		rectangle(frame, fullbody, Scalar(216, 235,52), 4, 8, 0);
	
		//upper body
		Rect upperbody;
		std::vector<Rect> ubodies;
		upperbody_cascade.detectMultiScale(frame_gray, ubodies, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE,
			Size(50, minBodyHeight * 0.5));
		for (int j = 0; j < ubodies.size(); j++)
		{
			Point center(ubodies[i].x + ubodies[i].width / 2, ubodies[i].y + ubodies[i].height / 2);
			Point first(center.x - (ubodies[i].width / 2), center.y - (ubodies[i].height / 2));
			Point second(center.x + (ubodies[i].width / 2), center.y + (ubodies[i].height / 2));
			upperbody = Rect(first, second);
			//rectangle(frame, upperbody, Scalar(255, 0, 255), 4, 8, 0);
			
			rectangle(frame, upperbody, Scalar(40, 100, 0), 4, 8, 0);
			//circle(frame, center, radius, Scalar(0, 255, 0), 4, 8, 0);
		}
		/////////////////////////////

		std::vector<Rect> lbodies;
		Rect lowerbody;
		lowerbody_cascade.detectMultiScale(frame_gray, lbodies, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE,
			Size(minBodyHeight * 0.3f, minBodyHeight * 0.5));


		for (int j = 0; j < lbodies.size(); j++) {
			Point center(lbodies[i].x + lbodies[i].width / 2, lbodies[i].y + lbodies[i].height / 2);
			Point first(center.x - (lbodies[i].width / 2), center.y - (lbodies[i].height / 2));
			Point second(center.x + (lbodies[i].width / 2), center.y + (lbodies[i].height / 2));
			lowerbody = Rect(first, second);
			//rectangle(frame, lowerbody, Scalar(0, 255, 255), 4, 8, 0);
			rectangle(frame, lbodies[j], Scalar(255, 0, 0), 4, 8, 0);
		}


		imshow(window_name, frame);
		waitKey();
	}
}



void body_detection() {
	String fullbody_cascade_name = "haarcascade_fullbody.xml";
	String lowerbody_cascade_name = "haarcascade_lowerbody.xml";
	String upperbody_cascade_name = "haarcascade_upperbody.xml";
	String mcs_upperbody_cascade_name = "haarcascade_mcs_upperbody.xml";

	// Load the cascades
	if (!fullbody_cascade.load(fullbody_cascade_name))
	{
		printf("Error loading face cascades !\n");
		return;
	}

	if (!lowerbody_cascade.load(lowerbody_cascade_name))
	{
		printf("Error loading eyes cascades !\n");
		return;
	}

	if (!upperbody_cascade.load(upperbody_cascade_name))
	{
		printf("Error loading eyes cascades !\n");
		return;
	}

	if (!mcs_upperbody_cascade.load(mcs_upperbody_cascade_name))
	{
		printf("Error loading eyes cascades !\n");
		return;
	}

	Mat src;
	Mat dst;
	char fname[MAX_PATH];

	while (openFileDlg(fname))
	{

		src = imread(fname, CV_LOAD_IMAGE_COLOR);
		dst = src.clone();
		int minFaceSize = 30;
		int minEyeSize = minFaceSize / 5; // conform proprietatilor antropomorfice ale fetei(idem pt.gura si nas)
		Fulldetection("Dst", dst, minFaceSize, minEyeSize);
	}
}

Rect FaceDetect(const string& window_name, Mat frame, int minFaceSize, int minEyeSize, bool hasFace, bool hasNose, bool hasEyes, bool hasMouth) {

	std::vector<Rect> faces;
	Mat grayFrame;
	cvtColor(frame, grayFrame, CV_BGR2GRAY);
	equalizeHist(grayFrame, grayFrame);

	face_cascade.detectMultiScale(grayFrame, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE,
		Size(minFaceSize, minFaceSize));
	Rect faceROI = faces[0];
	for (int i = 0; i < faces.size(); i++)
	{
		Point center(faces[i].x + faces[i].width * 0.5, faces[i].y + faces[i].height * 0.5);
		Point first(faces[i].x, faces[i].y);
		Point second(faces[i].x + faces[i].width, faces[i].y + faces[i].height);

		rectangle(frame, faces[0], Scalar(0, 0, 255));

		std::vector<Rect> eyes;
		Mat faceROI = grayFrame(faces[i]);
		eyes_cascade.detectMultiScale(faceROI, eyes, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE,
			Size(minEyeSize, minEyeSize));

		for (int j = 0; hasEyes && j < eyes.size(); j++)
		{
			Point center(faces[i].x + eyes[j].x + eyes[j].width * 0.5,
				faces[i].y + eyes[j].y + eyes[j].height * 0.5);
			int radius = cvRound((eyes[j].width + eyes[j].height) * 0.25);
			circle(frame, center, radius, Scalar(255, 0, 0), 4, 8, 0);
		}

		Rect rect_mouth;
		rect_mouth.x = faces[i].x + faces[i].width / 3;
		rect_mouth.y = faces[i].y + 0.65 * faces[i].height;
		rect_mouth.width = faces[i].width / 2;
		rect_mouth.height = 0.35 * faces[i].height;

		Mat mouth_ROI = grayFrame(rect_mouth);
		std::vector<Rect> mouth;
		int minMouthSize = 0.2 * minFaceSize;
		mouth_cascade.detectMultiScale(mouth_ROI, mouth, 1.1, 1, 0 | CV_HAAR_SCALE_IMAGE, Size(minMouthSize, minMouthSize));

		for (int j = 0; hasMouth && j < mouth.size(); j++)
		{
			Point center(rect_mouth.x + mouth[j].x + mouth[j].width * 0.5,
				rect_mouth.y + mouth[j].y + mouth[j].height * 0.5);
			int radius = cvRound((mouth[j].width + mouth[j].height) * 0.25);
			circle(frame, center, radius, Scalar(0, 255, 0), 4, 8, 0);
		}


		Rect rect_nose;
		rect_nose.x = faces[i].x + faces[i].width / 3;
		rect_nose.y = faces[i].y + 0.3 * faces[i].height;
		rect_nose.width = faces[i].width / 2;
		rect_nose.height = 0.5 * faces[i].height;

		Mat nose_ROI = grayFrame(rect_nose);
		std::vector<Rect> nose;
		int minNoseSize = 0.10 * minFaceSize;
		nose_cascade.detectMultiScale(nose_ROI, nose, 1.1, 1, 0 | CV_HAAR_SCALE_IMAGE, Size(minNoseSize, minNoseSize));

		for (int j = 0; hasNose && j < nose.size(); j++)
		{
			Point center(rect_nose.x + nose[j].x + nose[j].width * 0.5,
				rect_nose.y + nose[j].y + nose[j].height * 0.5);
			int radius = cvRound((nose[j].width + nose[j].height) * 0.25);
			circle(frame, center, radius, Scalar(0, 0, 255), 4, 8, 0);
		}
	}
	imshow(window_name, frame);
	return faceROI;
	waitKey();
}

void lab10() {

	String face_cascade_name = "haarcascade_frontalface_alt.xml";
	String eyes_cascade_name = "haarcascade_eye_tree_eyeglasses.xml";
	String mouth_cascade_name = "haarcascade_mcs_mouth.xml";
	String nose_cascade_name = "haarcascade_mcs_nose.xml";


	if (!face_cascade.load(face_cascade_name))
	{
		printf("Error loading face cascades !\n");
		return;
	}
	if (!eyes_cascade.load(eyes_cascade_name))
	{
		printf("Error loading eyes cascades !\n");
		return;
	}
	if (!mouth_cascade.load(mouth_cascade_name))
	{
		printf("Error loading mouth cascades !\n");
		return;
	}
	if (!nose_cascade.load(nose_cascade_name))
	{
		printf("Error loading nose cascades !\n");
		return;
	}
	VideoCapture cap("Videos/test_msv1_short.avi");
	if (!cap.isOpened()) {
		printf("Cannot open video capture device.\n");
		waitKey();
		return;
	}

	Mat frame, gray;
	Mat backgnd;
	Mat diff;
	Mat dst;
	char c;
	int frameNum = -1;

	cap.read(frame);
	const unsigned char Th = 25;
	const double alpha = 0.05;

	for (;;) {
		double t = (double)getTickCount();

		cap >> frame;
		if (frame.empty())
		{
			printf("End of video file\n");
			break;
		}
		++frameNum;
		if (frameNum == 0)
			imshow("src", frame);

		cvtColor(frame, gray, CV_BGR2GRAY);

		GaussianBlur(gray, gray, Size(5, 5), 0.8, 0.8);
		dst = Mat::zeros(gray.size(), gray.type());
		const int channels_gray = gray.channels();
		if (channels_gray > 1)
			return;
		if (frameNum > 0)
		{
			int minFaceSize = 50;
			int minEyeSize = minFaceSize / 5;

			Rect faceROI = FaceDetect("face", frame, minFaceSize, minEyeSize, true, false, false, false);
			absdiff(gray, backgnd, diff);
			backgnd = gray.clone();
			for (int i = 0; i < dst.rows; i++)
			{
				for (int j = 0; j < dst.cols; j++) {
					if (diff.at<uchar>(i, j) > Th)
					{
						dst.at<uchar>(i, j) = 255;
					}
				}
			}
			Mat element = getStructuringElement(MORPH_CROSS, Size(3, 3));
			Mat temp = dst(faceROI);
			imshow("background diff", backgnd);
			imshow("src", frame);

			erode(temp, temp, element, Point(-1, -1), 1);
			dilate(temp, temp, element, Point(-1, -1), 1);
			imshow("tempFinal", temp);

			typedef struct {
				double arie;
				double xc;
				double yc;
			} mylist;
			vector<mylist> candidates;
			candidates.clear();
			vector<vector<Point> > contours;
			vector<Vec4i> hierarchy;
			Mat roi = Mat::zeros(temp.rows, temp.cols, CV_8UC3);
			findContours(temp, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
			Moments m;
			if (contours.size() > 0)
			{
				int idx = 0;
				for (; idx >= 0; idx = hierarchy[idx][0])
				{
					const vector<Point>& c = contours[idx];
					m = moments(c);
					double arie = m.m00;
					double xc = m.m10 / m.m00;
					double yc = m.m01 / m.m00;
					Scalar color(rand() & 255, rand() & 255, rand() & 255);
					drawContours(roi, contours, idx, color, CV_FILLED, 8, hierarchy);
					mylist elem;
					elem.arie = arie;
					elem.xc = xc;
					elem.yc = yc;
					candidates.push_back(elem);
				}
			}
			if (candidates.size() >= 2)
			{
				mylist leftEye = candidates[0], rightEye = candidates[0];
				double arie1 = 0, arie2 = 0;
				for (mylist e : candidates)
				{
					if (e.arie > arie1)
					{
						arie2 = arie1;
						leftEye = rightEye;
						arie1 = e.arie;
						rightEye = e;
					}
					else
					{
						if (e.arie > arie2)
						{
							arie2 = e.arie;
							leftEye = e;
						}
					}
				}

				if ((abs(rightEye.yc - leftEye.yc) < 0.1 * faceROI.height && abs(rightEye.yc - leftEye.yc) < (faceROI.height) / 2))

					if (abs(rightEye.xc - leftEye.xc) > 0.3 * faceROI.width && abs(rightEye.xc - leftEye.xc) < 0.5 * faceROI.width)
						if (rightEye.xc - leftEye.xc > 0) {
							if (leftEye.xc <= (faceROI.width) / 2 && rightEye.xc >= (faceROI.width) / 2)
							{
								DrawCross(roi, Point(rightEye.xc, rightEye.yc), 15, Scalar(134, 21, 255), 2);
								DrawCross(roi, Point(leftEye.xc, leftEye.yc), 15, Scalar(0, 67, 0), 2);
								rectangle(frame, faceROI, Scalar(0, 255, 23));
								imshow("sursa", frame);
							}
						}
						else if (leftEye.xc >= (faceROI.width) / 2 && rightEye.xc <= (faceROI.width) / 2) {
							{
								DrawCross(roi, Point(leftEye.xc, leftEye.yc), 15, Scalar(0, 231, 255), 2);
								DrawCross(roi, Point(rightEye.xc, rightEye.yc), 15, Scalar(21, 255, 0), 2);
								rectangle(frame, faceROI, Scalar(24, 255, 0));
								imshow("sursa", frame);
							}
						}

			}
			imshow("colored", roi);
		}
		else
			backgnd = gray.clone();
		c = cvWaitKey(0);
		if (c == 27) {
			printf("ESC pressed - playback finished\n");
			break;
		}
		t = ((double)getTickCount() - t) / getTickFrequency();
		printf("%d - %.3f [ms]\n", frameNum, t * 1000);
	}
}

void laborator10() {

	String face_cascade_name = "haarcascade_frontalface_alt.xml";
	String eyes_cascade_name = "haarcascade_eye_tree_eyeglasses.xml";
	String mouth_cascade_name = "haarcascade_mcs_mouth.xml";
	String nose_cascade_name = "haarcascade_mcs_nose.xml";


	if (!face_cascade.load(face_cascade_name))
	{
		printf("Error loading face cascades !\n");
		return;
	}
	if (!eyes_cascade.load(eyes_cascade_name))
	{
		printf("Error loading eyes cascades !\n");
		return;
	}
	if (!mouth_cascade.load(mouth_cascade_name))
	{
		printf("Error loading mouth cascades !\n");
		return;
	}
	if (!nose_cascade.load(nose_cascade_name))
	{
		printf("Error loading nose cascades !\n");
		return;
	}
	VideoCapture cap("Videos/test_msv1_short.avi");
	if (!cap.isOpened()) {
		printf("Cannot open video capture device.\n");
		waitKey();
		return;
	}

	Mat frame, gray;
	Mat backgnd;
	Mat diff;
	Mat dst;
	char c;
	int frameNum = -1;

	cap.read(frame);
	const unsigned char Th = 25;
	const double alpha = 0.05;

	for (;;) {
		double t = (double)getTickCount();

		cap >> frame;
		if (frame.empty())
		{
			printf("End of video file\n");
			break;
		}
		++frameNum;
		if (frameNum == 0)
			imshow("src", frame);
		cvtColor(frame, gray, CV_BGR2GRAY);
		//GaussianBlur(gray, gray, Size(5, 5), 0.8, 0.8);
		dst = Mat::zeros(gray.size(), gray.type());
		const int channels_gray = gray.channels();
		if (channels_gray > 1)
			return;
		if (frameNum > 0)
		{
			int minFaceSize = 50;
			int minEyeSize = minFaceSize / 5;
			Rect faceROI = FaceDetect("FACE", frame, minFaceSize, minEyeSize, true, false, false, false);
			absdiff(gray, backgnd, diff);
			backgnd = gray.clone();

			for (int i = 0; i < dst.rows; i++)
			{
				for (int j = 0; j < dst.cols; j++) {
					if (diff.at<uchar>(i, j) > Th)
					{
						dst.at<uchar>(i, j) = 255;
					}
				}
			}
			Mat element = getStructuringElement(MORPH_CROSS, Size(3, 3));

			erode(diff, diff, element, Point(-1, -1), 1);
			dilate(diff, diff, element, Point(-1, -1), 1);
			Mat temp = dst(faceROI);
			imshow("diff", diff);

			/*erode(temp, temp, element, Point(-1, -1), 1);
			dilate(temp, temp, element, Point(-1, -1), 2);
			erode(temp, temp, element, Point(-1, -1), 1);
			imshow("tempFinal", temp);*/

			typedef struct {
				double arie;
				double xc;
				double yc;
			} mylist;
			vector<mylist> candidates;
			candidates.clear();
			vector<vector<Point> > contours;
			vector<Vec4i> hierarchy;
			Mat roi = Mat::zeros(temp.rows, temp.cols, CV_8UC3);
			findContours(temp, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
			Moments m;
			if (contours.size() > 0)
			{
				int idx = 0;
				for (; idx >= 0; idx = hierarchy[idx][0])
				{
					const vector<Point>& c = contours[idx];
					m = moments(c);
					double arie = m.m00;
					double xc = m.m10 / m.m00;
					double yc = m.m01 / m.m00;
					Scalar color(rand() & 255, rand() & 255, rand() & 255);
					drawContours(roi, contours, idx, color, CV_FILLED, 8, hierarchy);
					mylist elem;
					elem.arie = arie;
					elem.xc = xc;
					elem.yc = yc;
					candidates.push_back(elem);
				}
			}
			if (candidates.size() >= 2)
			{
				mylist leftEye = candidates[0], rightEye = candidates[0];
				double arie1 = 0, arie2 = 0;
				for (mylist e : candidates)
				{
					if (e.arie > arie1)
					{
						arie2 = arie1;
						leftEye = rightEye;
						arie1 = e.arie;
						rightEye = e;
					}
					else
					{
						if (e.arie > arie2)
						{
							arie2 = e.arie;
							leftEye = e;
						}
					}
				}

				if ((abs(rightEye.yc - leftEye.yc) < 0.1 * faceROI.height && abs(rightEye.yc - leftEye.yc) < (faceROI.height) / 2))

					if (abs(rightEye.xc - leftEye.xc) > 0.3 * faceROI.width && abs(rightEye.xc - leftEye.xc) < 0.5 * faceROI.width)
						if (rightEye.xc - leftEye.xc > 0) {
							if (leftEye.xc <= (faceROI.width) / 2 && rightEye.xc >= (faceROI.width) / 2)
							{
								rectangle(frame, faceROI, Scalar(0, 0, 255));
								imshow("sursa", frame);
							}
						}
						else if (leftEye.xc >= (faceROI.width) / 2 && rightEye.xc <= (faceROI.width) / 2) {
							{
								rectangle(frame, faceROI, Scalar(0, 255, 0));
								imshow("sursa", frame);
							}
						}
				DrawCross(roi, Point(rightEye.xc, rightEye.yc), 15, Scalar(0, 0, 255), 2);
				DrawCross(roi, Point(leftEye.xc, leftEye.yc), 15, Scalar(255, 0, 0), 2);
			}
			imshow("colored", roi);
		}
		else
			backgnd = gray.clone();
		c = cvWaitKey(0);
		if (c == 27) {
			printf("ESC pressed - playback finished\n");
			break;
		}
		t = ((double)getTickCount() - t) / getTickFrequency();
		printf("%d - %.3f [ms]\n", frameNum, t * 1000);
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
		printf(" 1 - Open image\n");
		printf(" 2 - Open BMP images from folder\n");
		printf(" 3 - Image negative - diblook style\n");
		printf(" 4 - BGR->HSV\n");
		printf(" 5 - Resize image\n");
		printf(" 6 - Canny edge detection\n");
		printf(" 7 - Edges in a video sequence\n");
		printf(" 8 - Snap frame from live video\n");
		printf(" 9 - Mouse callback demo\n");
		printf(" 10 - lab 3\n");
		printf(" 11 - lab3 - Build\n");
		printf(" 12 - lab 4 -Region Growing\n");
		printf(" 14 - lab 5 - Detect corners\n");
		printf(" 15 - lab 5 - Harris\n");
		printf(" 16 -lab 5 - Video -  detect corners\n");
		printf(" 17 -lab 6 - Video -  background substraction - metoda 1 \n");
		printf(" 18 -lab 6 - Video -  background substraction  - metoda 2 \n");
		printf(" 19 -lab 6 - Video -  background substraction - metoda 3 \n");
		printf(" 20 - lab 7 - Horn_Schunk \n ");
		printf(" 21 - lab 7 - OpticalFlow LK\n ");
		printf(" 22 - lab 8 - Middleburry\n");
		printf(" 23 - Face detection eyes \n");
		printf(" 24 - Face detection eyes mouth nose\n");
		printf(" 25 - Face detect video\n");
		printf(" 26 - Face detect LBP\n");
		printf(" 27 - Body detection\n");
		printf(" 28 - Laborator 10\n");
		printf(" 29 - Laborator 10*\n");
		printf(" 0 - Exit\n\n");
		printf("Option: ");
		scanf("%d",&op);
		switch (op)
		{
			case 1:
				testOpenImage();
				break;
			case 2:
				testOpenImagesFld();
				break;
			case 3:
				testParcurgereSimplaDiblookStyle(); //diblook style
				break;
			case 4:
				//testColor2Gray();
				testBGR2HSV();
				break;
			case 5:
				testResize();
				break;
			case 6:
				testCanny();
				break;
			case 7:
				testVideoSequence();
				break;
			case 8:
				testSnap();
				break;
			case 9:
				testMouseClick();
				break;
			case 10: 
				clasificare_pixeli();
				break;
			case 11:
				L3_ColorModel_Build();
				break;
			case 12:
				lab4();
				break;
			case 13:
				regionGrowing();
				break;
			case 14:
				lab5();
				break;
			case 15:
				cornerHarris_demo();
				break;
			case 16:
				testVideoCorners();
				break;
			case 17:
				testVideo(1);
				break;
			case 18:
				testVideo(2);
				break;
			case 19:
				testVideo(3);
				break;
			case 20:
				Horn_Schunk();
				break;
			case 21:
				calcOpticalFlowPyrLK();
				break;
			case 22:
				flux_optic();
				break;
			case 23:
				face_detection_eyes();
				break;
			case 24:
				face_detection_mouth_nose();
				break;
			case 25:
				face_detect_video();
				break;
			case 26:
				face_detection_lbp();
				break;
			case 27:
				body_detection();
				break;
			case 28:
				lab10();
				break;
			case 29:
				laborator10();
				break;

		}
	}
	while (op!=0);
	return 0;
}