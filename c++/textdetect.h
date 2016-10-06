#ifndef TEXTDETECT_H
#define TEXTDETECT_H

#include <opencv2/opencv.hpp>
#include <vector>

using cv::Mat;
using std::vector;

class DetectText {
	private:
		// --------------------------------------------		
		// Custom datatypes
		// --------------------------------------------	

		// Stores a pixel location and its stroke width value	
		struct SWTPoint2d {
			int x;
			int y;
			float SWT;
		};

		typedef std::pair<SWTPoint2d, SWTPoint2d> SWTPointPair2d;
		typedef std::pair<cv::Point, cv::Point>   SWTPointPair2i;

		struct Point2dFloat {
			float x;
			float y;
		};
		struct Point3dFloat {
			float x;
			float y;
			float z;
		};

		// Stores endpoints of SWT ray and a list of the points in between
		struct Ray {
			SWTPoint2d p;
			SWTPoint2d q;
			vector<SWTPoint2d> points;
		};
		// Stores all rays found in the SWT function
		vector<Ray> rays;

		// Stores original input image (screenshot)
		// and the SWT image, from its first pass through the median filtering
		Mat input;
		Mat SWTImage; 

		// For showing output of component filtering step and word chaing step
		// Not really necesary for final program
		Mat componentDisplay;
		Mat chainDisplay;

		// Used to hold bounding boxes around confirmed words
		vector<SWTPointPair2i> bb;

		// Stores user input about whether or not the text is dark on a light background
		bool dark_on_light;

		// Stores list of letters found in SWT
		vector<vector<SWTPoint2d> > components;

		// Stores a chain of letters (components) together
		// Carries IDs of letters on the ends, a list of IDs for the middle letters
		// Also contains a vector describing the direction of the chain
		struct Chain {
			int p;
			int q;
			float dist;
			bool merged;
			Point2dFloat direction;
			vector<int> components;
		};
		vector<Chain> chains;

		// SWT members
		Mat edgeImage;
		Mat gradientX;
		Mat gradientY;


		// --------------------------------------------
		// Stoke width transform and related functions
		// --------------------------------------------		
		void edgeAndGradient();
		void strokeWidthTransform();
		void SWTMedianFilter();
		void displaySWT();
		// Helper function
		static bool Point2dSort (const SWTPoint2d &lhs, const SWTPoint2d &rhs);


		// --------------------------------------------		
		// Connected components processing
		// --------------------------------------------		
		// Stores filtered list of letters
		vector<vector<SWTPoint2d> > validComponents;
		vector<SWTPointPair2d > compBB;
		vector<Point2dFloat> compCenters;
		vector<float> compMedians;
		vector<SWTPoint2d> compDimensions;
		void findLegallyConnectedComponents();
		void filterComponents();
		void renderComponentsWithBoxes();
		// Helper functions
		void renderComponents(Mat&);
		void componentStats(const std::vector<SWTPoint2d> & component, 
		float&, float&, float&, int&, int&, int&, int&);


		// --------------------------------------------		
		// Word chaining functions
		// --------------------------------------------		
		void makeChains();
		// Helper functions
		bool sharesOneEnd(Chain, Chain);
		static bool chainSortDist (const Chain &lhs, const Chain &rhs);
		static bool chainSortLength (const Chain &lhs, const Chain &rhs);


	public:
		void detectText();
		DetectText(char *argv[]);
};

#endif
