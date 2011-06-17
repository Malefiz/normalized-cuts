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
#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <ietl/interface/ublas.h>
#include <ietl/vectorspace.h>
#include <ietl/lanczos.h>
#include <boost/random.hpp>
#include <boost/tuple/tuple.hpp>
#include <boost/limits.hpp>
#include <cmath>
#include <limits>

using namespace std;
using namespace TCLAP;
using namespace cv;
using namespace boost::numeric;
using namespace boost;
using namespace ietl;

typedef ublas::compressed_matrix<double> UBMat;
typedef ublas::vector<double> UBVector;

inline double distanceAffinity(double x1, double y1, double x2, double y2,double scale)
{
	return -(x1-x2)*(x1-x2)+(y1-y2)*(y1-y2)/scale;

}

inline double intensityAffinity(double i1, double i2, double scale)
{
	return -(i1-i2)*(i1-i2)/scale;
}

template <typename T,int c>
inline double vecAffinity(const cv::Vec<T,c>& x,const cv::Vec<T,c>& y,double scale)
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

/**
 * @param A affinity matrix
 * @param img the image
 * @param scale1/2/3 the scale factor for the affinity calculation
 * @param sparsity_factor number of sample standard deviations to use as a cutoff criterion
 */
void createAffinityMatrix(Mat img, double scale1, double scale2, double scale3, UBMat& A, UBMat& N,  double sparsity_factor=0.1)
{
	int size=(size_t)img.rows*img.cols;
	Mat_<Vec<double,33> > filteredImg = filterImage(img,createFilterBank(11));
	A.resize(size,size);
	//dry run/ calculate sample standard deviation
	//for each pixel affinity to each other pixel
	double s2=0;
	double s3=0;
	double mean=0;
	for(int r=0;r<img.rows;r++)
	{
		for(int c=0;c<img.cols;c++)
		{
			for(int i=0;i<img.rows;i++)
			{
				for(int j=0;j<img.cols;j++)
				{
					double affinity=distanceAffinity(r,c,i,j,scale1);
					affinity+=vecAffinity<uchar,3>(img.at<Vec3b>(r,c),img.at<Vec3b>(i,j),scale2);
					affinity+=vecAffinity<double,33>(filteredImg(r,c),filteredImg(i,j),scale3);
					affinity=exp(affinity);
					s2+=affinity;
					s3+=affinity*affinity;
					mean+=affinity;
				}
			}
		}
	}
	mean/=size;
	double cutoff = sparsity_factor*sqrt(s2*(size*s3/s2-s2))/size;
	//actual affinity matrix construction
	for(int r=0;r<img.rows;r++)
	{
		for(int c=0;c<img.cols;c++)
		{
			for(int i=0;i<img.rows;i++)
			{
				for(int j=0;j<img.cols;j++)
				{
					double affinity=distanceAffinity(r,c,i,j,scale1);
					affinity+=vecAffinity<uchar,3>(img.at<Vec3b>(r,c),img.at<Vec3b>(i,j),scale2);
					affinity+=vecAffinity<double,33>(filteredImg(r,c),filteredImg(i,j),scale3);
					affinity=exp(affinity);
					if(fabs(mean-affinity)<cutoff)
						A(r*img.cols+c,i*img.cols+j)=affinity;
				}
			}
		}
	}
	//Create normalized affinity matrix
	UBMat D((size_t)img.rows*img.cols,(size_t)img.rows*img.cols);
	for(int i=0;i<size;i++)
	{
		double degree=0.0;
		for(int j=0;j<size;j++)
		{
			degree+=A(i,j);
		}
		D(i,i)=1.0/sqrt(degree);
	}
	N=prod(A,D);
	N=prod(D,N);
}


UBVector eigenSolve(UBMat& N)
{

	vectorspace<UBVector> vec(N.size1());
	lagged_fibonacci607 mygen;
	lanczos<UBMat,vectorspace<UBVector> > solver(N,vec);

	//First we compute the two lowest eigenvalues, the lowest is guaranteed to be 0
	//The second smallest is what we are looking for
	int max_iter = 10*N.size1();
	double rel_tol = 500*numeric_limits<double>::epsilon();
	double abs_tol = pow(numeric_limits<double>::epsilon(),2./3);
	int n_lowest_eigenval = 2;
	vector<double> eigen;
	vector<double> err;
	//std::vector<int> multiplicity;
	lanczos_iteration_nlowest<double> iter(max_iter,n_lowest_eigenval,rel_tol,abs_tol);

	solver.calculate_eigenvalues(iter,mygen);
	eigen = solver.eigenvalues();
	err = solver.errors();
	//multiplicity = solver.multiplicities();

	assert(eigen.at(0)<eigen.at(1));

	//Now we compute the eigenvector belonging to the second largest eigenvalue

	vector<UBVector> eigenvectors;
	Info<double> info;
	solver.eigenvectors(eigen.begin()+1,eigen.end(),back_inserter(eigenvectors),info,mygen);

	return eigenvectors.at(0);
}

double cut(UBMat& A, UBVector& v, double threshold)
{
	double cut=0.0;
	for(int i=0;i<v.size();i++)
	{
		bool lower=v(i)<threshold;
		for(int j=0;j<v.size();j++)
		{
			if(lower && v(j)>=threshold)
				cut+=A(i,j);
		}
	}
	return cut;
}

double assoc(UBMat& A, UBVector& v, double threshold,bool a)
{
	double cut=0.0;
	for(int i=0;i<v.size();i++)
	{
		bool lower;
		if(a)
			lower=v(i)<threshold;
		else
			lower=v(i)>=threshold;
		if(lower)
		{
			for(int j=0;j<v.size();j++)
			{
				cut+=A(i,j);
			}
		}
	}
	return cut;
}

inline void indexToRowCol(int index, int cols, int& row, int& col)
{
	row=index/cols;
	col=index/cols;
}

Mat_<uchar> getMask(Mat img, UBVector& v, UBMat& A)
{
	//find best threshold
	double best_threshold=0.0;
	double lowest_cost=DBL_MAX;
	for(int i=0;i<v.size();i++)
	{
		double threshold=v(i);
		double cutAB = cut(A,v,threshold);
		double assocAV = assoc(A,v,threshold,true);
		double assocBV = assoc(A,v,threshold,false);
		double cost = cutAB/assocAV + cutAB/assocBV;
		if(cost<lowest_cost)
		{
			lowest_cost=cost;
			best_threshold=threshold;
		}
	}
	Mat_<uchar> mask(img.rows,img.cols);
	//create Mask
	for(int i=0;i<v.size();i++)
	{
		int r,c;
		indexToRowCol(i,img.cols,r,c);
		if(v(i)<best_threshold)
		{
			mask(r,c)=1;
		}
		else
		{
			mask(r,c)=0;
		}
	}
	return mask;
}

tuple<Mat,Mat> segment(Mat_<uchar> mask, Mat img)
{
	Mat A(img.size().width,img.size().height,CV_8UC3);
	Mat B(img.size().width,img.size().height,CV_8UC3);

	for(int i=0;i<mask.size().width;i++)
	{
		for(int j=0;j<mask.size().height;j++)
		{
			if(mask(i,j)==0)
			{
				A.at<Vec3b>(i,j)=Vec3b(0,0,0);
			}else
			{
				A.at<Vec3b>(i,j)=img.at<Vec3b>(i,j);
			}
		}
	}


	Mat_<uchar> inv_mask(mask.size().width,mask.size().height);
	for(int i=0;i<mask.size().width;i++)
	{
		for(int j=0;j<mask.size().height;j++)
		{
			inv_mask(i,j)=1-mask(i,j);
		}
	}

	for(int i=0;i<inv_mask.size().width;i++)
	{
		for(int j=0;j<inv_mask.size().height;j++)
		{
			if(inv_mask(i,j)==0)
			{
				B.at<Vec3b>(i,j)=Vec3b(0,0,0);
			}else
			{
				B.at<Vec3b>(i,j)=img.at<Vec3b>(i,j);
			}
		}
	}

	return tuple<Mat,Mat>(A,B);
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
