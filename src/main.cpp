/*
 * main.cpp
 *
 *  Created on: 03.06.2011
 *      Author: flo
 */

#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <tclap/CmdLine.h>
#include <cmath>
#include <boost/numeric/ublas/symmetric.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <ietl/interface/ublas.h>
#include <ietl/vectorspace.h>
#include <ietl/lanczos.h>
#include <boost/random.hpp>
#include <boost/limits.hpp>
#include <cmath>
#include <limits>

using namespace std;
using namespace TCLAP;
using namespace cv;

typedef boost::numeric::ublas::symmetric_matrix<double, boost::numeric::ublas::lower> UBMat;
typedef boost::numeric::ublas::vector<double> UBVector;

double distanceAffinity(double x1, double y1, double x2, double y2,double scale)
{
	return -(x1-x2)*(x1-x2)+(y1-y2)*(y1-y2)/scale;

}

double intensityAffinity(double i1, double i2, double scale)
{
	return -(i1-i2)*(i1-i2)/scale;
}

double colorAffinity(Vec3b& x, Vec3b& y,double scale)
{
	return - (x-y).dot(x-y)/scale;
}

Mat getGaussianKernel2D(double sigma1, double sigma2, double angle, int size)
{
	assert(size%2==1);
	Mat rot = getRotationMatrix2D(Point2f(size/2+1.0,size/2+1.0),angle,1.0);
	Mat sigma(3,2,CV_64FC1);
	sigma.at<double>(0,0)=1.0/sigma1;
	sigma.at<double>(0,1)=0;
	sigma.at<double>(1,0)=0;
	sigma.at<double>(1,1)=1.0/sigma2;
	sigma.at<double>(2,0)=0.0;
	sigma.at<double>(2,1)=0.0;

	Mat invsigma=rot*sigma;
	double factor = 1.0/(2.0*M_PI*sqrt(sigma1*sigma2));
	Mat mu(2,1,CV_64FC1);
	mu.at<double>(0)=size/2+1.0;
	mu.at<double>(1)=size/2+1.0;

	Mat kernel(size,size,CV_64FC1);
	Mat x(2,1,CV_64FC1);
	for(int r=0;r<size;r++)
	{
		double * row=kernel.ptr<double>(r);
		for(int c=0;c<size;c++)
		{
			x.at<double>(0)=r;
			x.at<double>(1)=c;
			Mat tmp=invsigma*(x-mu);
			double m=(x-mu).dot(tmp);
			row[c]=factor*exp(-0.5*m);
		}
	}
	return kernel;
}

Mat getDoGKernel2D(double sigma1, double sigma2, double angle,int size,double K)
{
	return getGaussianKernel2D(sigma1,sigma2,angle,size)-getGaussianKernel2D(K*sigma1,K*sigma2,angle,size);
}


vector<Ptr<FilterEngine> > createFilterBank(int size)
{
	vector<Ptr<FilterEngine> > bank;
	//Point filters
	bank.push_back(createLinearFilter(CV_64FC3,CV_64FC3,getGaussianKernel2D(0.2,0.2,0.0,size)));
	bank.push_back(createLinearFilter(CV_64FC3,CV_64FC3,getGaussianKernel2D(0.8,0.8,0.0,size)));
	bank.push_back(createLinearFilter(CV_64FC3,CV_64FC3,getGaussianKernel2D(1.2,1.2,0.0,size)));
	bank.push_back(createLinearFilter(CV_64FC3,CV_64FC3,getGaussianKernel2D(2.,2.,0.0,size)));
	bank.push_back(createLinearFilter(CV_64FC3,CV_64FC3,getGaussianKernel2D(3.,3.,0.0,size)));

	bank.push_back(createLinearFilter(CV_64FC3,CV_64FC3,getGaussianKernel2D(3.0,0.4,0.0,size)));
	bank.push_back(createLinearFilter(CV_64FC3,CV_64FC3,getGaussianKernel2D(3.0,0.4,30.0,size)));
	bank.push_back(createLinearFilter(CV_64FC3,CV_64FC3,getGaussianKernel2D(3.0,0.4,60.0,size)));
	bank.push_back(createLinearFilter(CV_64FC3,CV_64FC3,getGaussianKernel2D(3.0,0.4,90.0,size)));
	bank.push_back(createLinearFilter(CV_64FC3,CV_64FC3,getGaussianKernel2D(3.0,0.4,120.0,size)));
	bank.push_back(createLinearFilter(CV_64FC3,CV_64FC3,getGaussianKernel2D(3.0,0.4,150.0,size)));

	return bank;

}

Mat_<Vec<double,33> > filterImage(Mat img, vector<Ptr<FilterEngine> > filterBank)
{
	assert(filterBank.size()==11);
	assert(img.type()==CV_64FC3);
	assert(img.channels()==3);
	Mat_<Vec<double,33> > filteredAll(img.rows,img.cols);

	namedWindow("Filtered",WINDOW_AUTOSIZE);

	int f=0;
	Mat filtered(img.rows,img.cols,img.type());
	for(vector<Ptr<FilterEngine> >::const_iterator iter=filterBank.begin();
			iter!=filterBank.end();
			iter++,f++
			)
	{
		Ptr<FilterEngine> filter = *iter;

		filter->apply(img,filtered);
		imshow("Filtered",filtered);
		for(int r=0;r<img.rows;r++)
		{
			double * filteredRow = filtered.ptr<double>(r);
			for(int c=0;c<img.cols;c++)
			{
				for(int channel=0;channel<3;channel++)
				{
					filteredAll(r,c).val[f*3+channel]=filteredRow[c*3 + channel];
				}
			}
		}
	}
	destroyWindow("Filtered");
	return filteredAll;

}

void createAffinityMatrix(UBMat& A, Mat img, double sparsity_factor=0.8)
{
	Mat filteredImg = filterImage(img,createFilterBank(11));
	A.resize((size_t)img.rows*img.cols,(size_t)img.rows*img.cols);
	for(int r=0;r<img.rows;r++)
	{
		for(int c=0;c<img.cols;c++)
		{

		}
	}
}

int main(int argc, char ** argv)
{
	CmdLine cmdLine("Normalized cut demo");

	ValueArg<string> imgArg("i","image","the image to be segmented",true,"","string");
	ValueArg<unsigned int> segArg("n","segments","the number of segments",false,2,"int >= 2");
	ValueArg<double> sigmaArg("s","scale","scale parameter for affinity matrix construction",false,1.0,"double");

	cmdLine.add(imgArg);
	cmdLine.add(segArg);
	cmdLine.add(sigmaArg);
	cmdLine.parse(argc,argv);
	cout<<imgArg.getValue()<<endl;
	Mat_<Vec3b> img = imread(imgArg.getValue().c_str());
	//assert(img.depth()==IPL_DEPTH_8U && img.channels()==3);

	Mat_<Vec3f> lab(img.rows,img.cols);

	//convert image to a uniform color space
	cvtColor(img,lab,CV_RGB2Lab);
	const char * hOrig ="Original";
	const char * hGrey= "LAB";
	namedWindow(hOrig,CV_WINDOW_AUTOSIZE);
	namedWindow(hGrey,CV_WINDOW_AUTOSIZE);

	imshow(hOrig,img);
	imshow(hGrey,lab);

	waitKey(0);

	destroyWindow(hOrig);
	destroyWindow(hGrey);
}
