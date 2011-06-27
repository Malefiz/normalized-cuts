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

typedef double FP_Type;
typedef Vec3d Point_Type;


typedef ublas::compressed_matrix<FP_Type> UBMat;
typedef ublas::vector<FP_Type> UBVector;



inline FP_Type distanceAffinity(FP_Type x1, FP_Type y1, FP_Type x2, FP_Type y2,FP_Type scale)
{
	return -((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2))/scale;

}

inline FP_Type intensityAffinity(FP_Type i1, FP_Type i2, FP_Type scale)
{
	return -(i1-i2)*(i1-i2)/scale;
}

template <typename T,int c>
inline FP_Type vecAffinity(const cv::Vec<T,c>& x,const cv::Vec<T,c>& y,FP_Type scale)
{
	return - (x-y).dot(x-y)/scale;
}


Mat_<FP_Type> getGaussianKernel2D(FP_Type sigma1, FP_Type sigma2, FP_Type angle, int size)
{
	assert(size%2==1);
	Mat_<FP_Type> rot = getRotationMatrix2D(Point2f(size/2+1.0,size/2+1.0),angle,1.0);
	Mat_<FP_Type> sigma(3,2);
	sigma(0,0)=1.0/sigma1;
	sigma(0,1)=0;
	sigma(1,0)=0;
	sigma(1,1)=1.0/sigma2;
	sigma(2,0)=0.0;
	sigma(2,1)=0.0;

	Mat_<FP_Type> invsigma=rot*sigma;
	FP_Type factor = 1.0/(2.0*M_PI*sqrt(sigma1*sigma2));
	Mat_<FP_Type> mu(2,1);
	mu(0)=size/2+1.0;
	mu(1)=size/2+1.0;

	Mat_<FP_Type> kernel(size,size);
	Mat_<FP_Type> x(2,1);
	for(int r=0;r<size;r++)
	{
		FP_Type * row=(FP_Type*)kernel.ptr(r);
		for(int c=0;c<size;c++)
		{
			x(0)=r;
			x(1)=c;
			Mat_<FP_Type> tmp=invsigma*(x-mu);
			FP_Type m=(x-mu).dot(tmp);
			row[c]=factor*exp(-0.5*m);
		}
	}
	return kernel;
}

Mat_<FP_Type> getDoGKernel2D(FP_Type sigma1, FP_Type sigma2, FP_Type angle,int size,FP_Type K)
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

Mat_<Vec<FP_Type,33> > filterImage(Mat_<Point_Type> img, vector<Ptr<FilterEngine> > filterBank)
{
	assert(filterBank.size()==11);
	assert(img.channels()==3);
	Mat_<Vec<FP_Type,33> > filteredAll(img.rows,img.cols);

	namedWindow("Filtered",WINDOW_AUTOSIZE);

	int f=0;
	Mat_<Point_Type> filtered(img.rows,img.cols);
	for(vector<Ptr<FilterEngine> >::const_iterator iter=filterBank.begin();
			iter!=filterBank.end();
			iter++,f++
			)
	{
		Ptr<FilterEngine> filter = *iter;
		Mat_<Point_Type> myimg(2,2);
		const Mat& myotherimg=myimg;
		CV_Assert( img.type() == myimg.type());
		filter->apply(img,filtered);
		imshow("Filtered",filtered);
		for(int r=0;r<img.rows;r++)
		{
			FP_Type * filteredRow = (FP_Type*)filtered.ptr(r);
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

struct AdjListEntry{
public:
	int j;
	FP_Type a;
};

struct AdjList{
	AdjListEntry * entries;
	int size;
};

/**
 * @param A affinity matrix
 * @param img the image
 * @param scale1/2/3 the scale factor for the affinity calculation
 * @param sparsity_factor number of sample standard deviations to use as a cutoff criterion
 */
void createAffinityMatrix(Mat_<Point_Type>  img, FP_Type scale1, FP_Type scale2, FP_Type scale3, UBMat& A, UBMat& N, UBMat& D,  FP_Type sparsity_factor=0.01)
{
	int size=(size_t)img.rows*img.cols;
	Mat_<Vec<FP_Type,33> > filteredImg = filterImage(img,createFilterBank(11));
	A.resize(size,size,false);
	N.resize(size,size,false);
	D.resize(size,size,false);
	//Construct an adjacency list
	AdjList * adjList=new AdjListEntry[size];
	for(int r=0;r<img.rows;r++)
	{
		for(int c=0;c<img.cols;c++)
		{
			int adjListSize=(max(0,c+5)-max(img.cols,c+5))*(max(0,r+5)-max(img.rows,r+5));
			AdjListEntry * adjListEntry=new AdjListEntry[adjListSize];
			adjList[r*img.cols+c].entries=adjListEntry;
			adjList[r*img.cols+c].size=adjListSize;
			int k=0;
			for(int i=max(0,r-5);i<min(img.rows,r+5);i++)
			{
				for(int j=max(0,c-5);j<min(img.cols,c+5);j++)
				{
					FP_Type affinity=distanceAffinity(r,c,i,j,scale1);
					affinity+=vecAffinity<uchar,3>(img(r,c),img(i,j),scale2);
					affinity+=vecAffinity<FP_Type,33>(filteredImg(r,c),filteredImg(i,j),scale3);
					adjListEntry[k++].a=exp(affinity);
					adjListEntry[k++].j=i*img.cols+j;
				}
			}
		}
	}
	//Create normalized affinity matrix
	double d[size];
	for(int i=0;i<size;i++)
	{
		FP_Type degree=0.0;
		for(int j=0;j<adjList[i].size;j++)
		{
			A(i,adjList[i]->entries[j].j)=adjList[i]->entries[j].a;
			degree+=adjList[i]->entries[j].a;
		}
		d[i]=1.0/sqrt(degree+DBL_EPSILON);
		D(i,i)=d[i];
	}
	//Right multiplication with D^(-1/2)
	for(int i=0;i<size;i++)
	{
		int row=i%img.cols;
		for(int j=0;j<adjList[i].size;j++)
		{
			adjList[i].entries[j].a*=d[adjList->entries[i].j];
		}
	}
	//Left multiplication with D^(-1/2)
	for(int i=0;i<size;i++)
	{
		int row=i%img.cols;
		for(int j=0;j<adjList[i].size;j++)
		{
			N(i,adjList[i]->entries[j].j)=adjList[i].entries[j].a*d[i];
		}
	}
}


UBVector eigenSolve(UBMat& N)
{

	vectorspace<UBVector> vec(N.size1());
	lagged_fibonacci607 mygen;
	lanczos<UBMat,vectorspace<UBVector> > solver(N,vec);

	//First we compute the two lowest eigenvalues, the lowest is guaranteed to be 0
	//The second smallest is what we are looking for
	int max_iter = N.size1();
	FP_Type rel_tol = 50000*numeric_limits<FP_Type>::epsilon();
	FP_Type abs_tol = pow(numeric_limits<FP_Type>::epsilon(),20./3);
	int n_lowest_eigenval = 2;
	vector<FP_Type> eigen;
	vector<FP_Type> err;
	//std::vector<int> multiplicity;
	lanczos_iteration_nlowest<FP_Type> iter(max_iter,n_lowest_eigenval,rel_tol,abs_tol);

	solver.calculate_eigenvalues(iter,mygen);
	eigen = solver.eigenvalues();
	err = solver.errors();
	//multiplicity = solver.multiplicities();

	assert(eigen.at(0)<eigen.at(1));

	//Now we compute the eigenvector belonging to the second largest eigenvalue

	vector<UBVector> eigenvectors;
	Info<FP_Type> info;
	solver.eigenvectors(eigen.begin()+1,eigen.end(),back_inserter(eigenvectors),info,mygen);

	return eigenvectors.at(0);
}

FP_Type cut(UBMat& A, UBVector& v, FP_Type threshold)
{
	FP_Type cut=0.0;
	for(size_t i=0;i<v.size();i++)
	{
		bool lower=v(i)<threshold;
		for(size_t j=0;j<v.size();j++)
		{
			if(lower && v(j)>=threshold)
				cut+=A(i,j);
		}
	}
	return cut;
}

FP_Type assoc(UBMat& A, UBVector& v, FP_Type threshold,bool a)
{
	FP_Type cut=0.0;
	for(size_t i=0;i<v.size();i++)
	{
		bool lower;
		if(a)
			lower=v(i)<threshold;
		else
			lower=v(i)>=threshold;
		if(lower)
		{
			for(size_t j=0;j<v.size();j++)
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

Mat_<uchar> getMask(Mat_<Point_Type> img, UBVector& v, UBMat& A)
{
	//find best threshold
	FP_Type best_threshold=0.0;
	FP_Type lowest_cost=FLT_MAX;
	for(size_t i=0;i<v.size();i++)
	{
		FP_Type threshold=v(i);
		FP_Type cutAB = cut(A,v,threshold);
		FP_Type assocAV = assoc(A,v,threshold,true);
		FP_Type assocBV = assoc(A,v,threshold,false);
		FP_Type cost = cutAB/assocAV + cutAB/assocBV;
		if(cost<lowest_cost)
		{
			lowest_cost=cost;
			best_threshold=threshold;
		}
	}
	Mat_<uchar> mask(img.rows,img.cols);
	//create Mask
	for(size_t i=0;i<v.size();i++)
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

tuple<Mat_<Point_Type>,Mat_<Point_Type> > segment(Mat_<uchar> mask, Mat_<Point_Type> img)
{
	Mat_<Point_Type> A(img.size().width,img.size().height);
	Mat_<Point_Type> B(img.size().width,img.size().height);

	for(int i=0;i<mask.size().width;i++)
	{
		for(int j=0;j<mask.size().height;j++)
		{
			if(mask(i,j)==0)
			{
				A(i,j)=Point_Type(0,0,0);
			}else
			{
				A(i,j)=img(i,j);
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
				B(i,j)=Point_Type(0,0,0);
			}else
			{
				B(i,j)=img(i,j);
			}
		}
	}

	return tuple<Mat_<Point_Type>,Mat_<Point_Type> >(A,B);
}

int main(int argc, char ** argv)
{
	CmdLine cmdLine("Normalized cut demo");

	ValueArg<string> imgArg("i","image","the image to be segmented",true,"","string");
	ValueArg<unsigned int> segArg("n","segments","the number of segments",false,2,"int >= 2");
	ValueArg<FP_Type> sigmaArg("s","scale","scale parameter for affinity matrix construction",false,1.0,"decimal");

	cmdLine.add(imgArg);
	cmdLine.add(segArg);
	cmdLine.add(sigmaArg);
	cmdLine.parse(argc,argv);
	cout<<imgArg.getValue()<<endl;
	Mat img = imread(imgArg.getValue().c_str());

	//assert(img.depth()==IPL_DEPTH_8U && img.channels()==3);

	Mat lab(img.rows,img.cols,img.type());
	//convert image to a uniform color space
	cvtColor(img,lab,CV_RGB2Lab);
	lab.convertTo(lab,CV_64FC3);

	const char * hOrig ="Original";
	const char * hGrey= "LAB";
	namedWindow(hOrig,CV_WINDOW_AUTOSIZE);
	namedWindow(hGrey,CV_WINDOW_AUTOSIZE);

	imshow(hOrig,img);
	imshow(hGrey,lab);

	UBMat A;//Affinity matrix
	UBMat N;//Normalized affinity matrix
	UBMat D;//Actually, its D^(-1/2)
	createAffinityMatrix(lab,1.0,1.0,1.0,A,N,D,-30);
	UBVector y=prod(D,eigenSolve(N));
	Mat_<uchar> mask = getMask(lab,y,A);
	tuple<Mat_<Point_Type>,Mat_<Point_Type> > segmented = segment(mask,img);

	const char * hSegment1="Segment1";
	const char * hSegment2="Segment2";
	namedWindow(hSegment1,CV_WINDOW_AUTOSIZE);
	namedWindow(hSegment2,CV_WINDOW_AUTOSIZE);

	imshow(hSegment1,segmented.get<0>());
	imshow(hSegment2,segmented.get<1>());

	waitKey(0);

	destroyWindow(hOrig);
	destroyWindow(hGrey);
	destroyWindow(hSegment1);
	destroyWindow(hSegment2);

}
