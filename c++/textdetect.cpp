#include "textdetect.h"

#include <iostream>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/connected_components.hpp>
#include <boost/unordered_map.hpp>
#include <boost/graph/floyd_warshall_shortest.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>

#define PI 3.14159265

using std::cout;
using std::cerr;
using std::endl;

using cv::Point2i;
using cv::Scalar;

// For displaying components
const Scalar BLUE (255, 0, 0);
const Scalar GREEN(0, 255, 0);
const Scalar RED  (0, 0, 255);


// Constructor
// Loads user-specified input image
// (Will eventually take a screenshot instead)
// TODO: Import bitmap instead of png
DetectText::DetectText(char *argv[]) {
	input = cv::imread(argv[1]);
	if (input.empty())
		cerr << "couldn't load query image" << endl;
	cv::cvtColor(input, input, CV_BGR2RGB);

	dark_on_light = atoi(argv[3]);
}


// Runs Canny edge detection on the image
// Finds x and y derivatives (gradient is necessary for SWT)
void DetectText::edgeAndGradient() {
	// Convert input to grayscale
	Mat grayImage(input.size(), CV_8UC1 );
	cv::cvtColor(input, grayImage, CV_RGB2GRAY);

	// Used for component display later
	componentDisplay = input.clone();
	chainDisplay = input.clone();

	// TODO: Trt up-scaling the image first
	// to improve separation of letters in Canny image

	// Create Canny Image
	double threshold_low = 175; 
	double threshold_high = 320; 
	cv::Canny(grayImage, edgeImage, threshold_low, threshold_high, 3); 
	cv::imwrite ("canny.png", edgeImage);

	// Create gradient X, gradient Y
	Mat gaussianImage(input.size(), CV_32FC1);
	grayImage.convertTo(gaussianImage, CV_32FC1, 1./255.);
	cv::GaussianBlur(gaussianImage, gaussianImage, cv::Size(5, 5), 0); 
	cv::Scharr(gaussianImage, gradientX, -1, 1, 0); 
	cv::Scharr(gaussianImage, gradientY, -1, 0, 1); 
	cv::GaussianBlur(gradientX, gradientX, cv::Size(3, 3), 0); 
	cv::GaussianBlur(gradientY, gradientY, cv::Size(3, 3), 0); 

	// Calculate SWT and return ray vectors
	SWTImage = input.clone();
	SWTImage.convertTo(SWTImage,CV_32FC1); 
	for(int row = 0; row < input.rows; row++) {
		float* ptr = (float*)SWTImage.ptr(row);
		for (int col = 0; col < input.cols; col++)
			*ptr++ = -1; 
	}
}


// First pass of the SWT
void DetectText::strokeWidthTransform() {
    // Step size for moving along the gradient
    float prec = .05;
	// Loop through every pixel in the image
    for( int row = 0; row < edgeImage.rows; row++ ){
        const uchar* ptr = (const uchar*)edgeImage.ptr(row);
        for ( int col = 0; col < edgeImage.cols; col++ ){
            // For all edge pixels
            if (*ptr > 0) {
				// Start ray at that pixel                
				Ray r;
                SWTPoint2d p;
                p.x = col;
                p.y = row;
                r.p = p;
                vector<SWTPoint2d> points;
                points.push_back(p);

                // Start traversal at the center of the initial pixel
				// and find the gradient there
                float curX = (float)col + 0.5;
                float curY = (float)row + 0.5;
                int curPixX = col;
                int curPixY = row;
                float G_x = gradientX.at<float>(row, col);
                float G_y = gradientY.at<float>(row, col);

                // Normalize the gradient
                // (Convert to unit length and to point into the letter stroke)
                float mag = sqrt( (G_x * G_x) + (G_y * G_y) );
                if (dark_on_light){
                    G_x = -G_x/mag;
                    G_y = -G_y/mag;
                } else {
                    G_x = G_x/mag;
                    G_y = G_y/mag;
                }

				
                while (true) {
                    curX += G_x*prec;
                    curY += G_y*prec;
					// Prevents calculation of SWT for a pixel multiple times
                    if ((int)(floor(curX)) != curPixX || (int)(floor(curY)) != curPixY) {
                        curPixX = (int)(floor(curX));
                        curPixY = (int)(floor(curY));

                        // Check if pixel is outside boundary of image
                        if (curPixX < 0 || (curPixX >= SWTImage.cols) || curPixY < 0 || (curPixY >= SWTImage.rows)) {
                            break;
						}

						// Adds this new pixel to the list of those in the ray for base pixel p
                        SWTPoint2d pnew;
                        pnew.x = curPixX;
                        pnew.y = curPixY;
                        points.push_back(pnew);

						// Loop until we've reached another edge point
                        if (edgeImage.at<uchar>(curPixY, curPixX) > 0) {
                            r.q = pnew;              
							// Normalize gradient at other end of ray
							// TODO: Do we need to normalize to test orthogonality?
                            float G_xt = gradientX.at<float>(curPixY,curPixX);
                            float G_yt = gradientY.at<float>(curPixY,curPixX);
                            mag = sqrt( (G_xt * G_xt) + (G_yt * G_yt) );
                            if (dark_on_light) {
                                G_xt = -G_xt / mag;
                                G_yt = -G_yt / mag;
                            } else {
                                G_xt = G_xt / mag;
                                G_yt = G_yt / mag;
                            }

							// Check to see if the gradients are orthogonal (using dot product)
                            // This ensures that the two edges are parallel, which they should be for letters
                            if (acos(G_x * -G_xt + G_y * -G_yt) < PI/2.0 ) {
								// Store the length of the ray in each of the ray's elements (or the min if there's already a value there)
                                float length = sqrt( ((float)r.q.x - (float)r.p.x)*((float)r.q.x - (float)r.p.x) + ((float)r.q.y - (float)r.p.y)*((float)r.q.y - (float)r.p.y));
                                for (vector<SWTPoint2d>::iterator pit = points.begin(); pit != points.end(); pit++) {
                                    // Stores stroke width value if none was there previously
                                    // If there's already a value, only replace it if the new value is smaller
                                    if (SWTImage.at<float>(pit->y, pit->x) < 0) {
                                        SWTImage.at<float>(pit->y, pit->x) = length;
                                    } else {
                                        SWTImage.at<float>(pit->y, pit->x) = std::min(length, SWTImage.at<float>(pit->y, pit->x));
                                    }
                                }
								// Add this ray to the rays vector
                                r.points = points;
                                rays.push_back(r);
                            }
                            break;
                        }
                    }
                }
            }
            ptr++;
        }
    }
}


// Function for sorting a ray's points by stroke width value
// Used for finding the median of the ray's values
bool DetectText::Point2dSort (const SWTPoint2d &lhs, const SWTPoint2d &rhs) {
    return lhs.SWT < rhs.SWT;
}


void DetectText::SWTMedianFilter() {
	// Sets points in a ray greater than the median of the values in that ray to this median
	// Accounts for discrepancies caused by corners of stokes
    for (auto& rit : rays) {
        for (auto& pit : rit.points) {
            pit.SWT = SWTImage.at<float>(pit.y, pit.x);
        }
        std::sort(rit.points.begin(), rit.points.end(), &Point2dSort);
        float median = (rit.points[rit.points.size()/2]).SWT;
        for (auto& pit : rit.points) {
            SWTImage.at<float>(pit.y, pit.x) = std::min(pit.SWT, median);
        }
	}
}


// Renders the SWT matrix into an image
// Not necessary for the final program
void DetectText::displaySWT() {
    Mat output( input.size(), CV_32FC1 );

	// Finds the largest and smallest stroke widths found
    float maxVal = 0;
    float minVal = 1e100;
    for (int row = 0; row < input.rows; row++) {
        const float* ptr = (const float*)input.ptr(row);
        for (int col = 0; col < input.cols; col++) {
            if (*ptr < 0) { }
            else {
                maxVal = std::max(*ptr, maxVal);
                minVal = std::min(*ptr, minVal);
            }
            ptr++;
        }
    }

	// Sets non-stroke width pixels to white and scales other to be between 0 and 1
    float difference = maxVal - minVal;
    for (int row = 0; row < input.rows; row++) {
        const float* ptrin = (const float*)input.ptr(row);
        float* ptrout = (float*)output.ptr(row);
        for (int col = 0; col < input.cols; col++) {
            if (*ptrin < 0) {
                *ptrout = 1;
            } else {
                *ptrout = ((*ptrin) - minVal)/difference;
            }
            ptrout++;
            ptrin++;
        }
    }

	// Scales back to a 0-255 range for rendering
    Mat saveSWT(input.size(), CV_8UC1);
    output.convertTo(saveSWT, CV_8UC1, 255);
    imwrite ("SWT.png", saveSWT);
}


// Finds stroke width pixels that are connected (diagonal pixels count)
// 
void DetectText::findLegallyConnectedComponents () {
	boost::unordered_map<int, int> map;
    boost::unordered_map<int, DetectText::SWTPoint2d> revmap;

    typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS> Graph;
    int num_vertices = 0;
    // Number vertices for graph. Associates each point with number
    for( int row = 0; row < SWTImage.rows; row++ ){
        float * ptr = (float*)SWTImage.ptr(row);
        for (int col = 0; col < SWTImage.cols; col++ ){
            if (*ptr > 0) {
                map[row * SWTImage.cols + col] = num_vertices;
                DetectText::SWTPoint2d p;
                p.x = col;
                p.y = row;
                revmap[num_vertices] = p;
                num_vertices++;
            }
            ptr++;
        }
    }

    Graph g(num_vertices);

    for( int row = 0; row < SWTImage.rows; row++ ){
        float * ptr = (float*)SWTImage.ptr(row);
        for (int col = 0; col < SWTImage.cols; col++ ){
            if (*ptr > 0) {
                // Check pixel to the right, right-down, down, left-down
				// Pixels must have roughly the same stroke width value
                int this_pixel = map[row * SWTImage.cols + col];
                if (col+1 < SWTImage.cols) {
                    float right = SWTImage.at<float>(row, col+1);
                    if (right > 0 && ((*ptr)/right <= 3.0 || right/(*ptr) <= 3.0))
                        boost::add_edge(this_pixel, map.at(row * SWTImage.cols + col + 1), g);
                }
                if (row+1 < SWTImage.rows) {
                    if (col+1 < SWTImage.cols) {
                        float right_down = SWTImage.at<float>(row+1, col+1);
                        if (right_down > 0 && ((*ptr)/right_down <= 3.0 || right_down/(*ptr) <= 3.0))
                            boost::add_edge(this_pixel, map.at((row+1) * SWTImage.cols + col + 1), g);
                    }
                    float down = SWTImage.at<float>(row+1, col);
                    if (down > 0 && ((*ptr)/down <= 3.0 || down/(*ptr) <= 3.0))
                        boost::add_edge(this_pixel, map.at((row+1) * SWTImage.cols + col), g);
                    if (col-1 >= 0) {
                        float left_down = SWTImage.at<float>(row+1, col-1);
                        if (left_down > 0 && ((*ptr)/left_down <= 3.0 || left_down/(*ptr) <= 3.0))
                            boost::add_edge(this_pixel, map.at((row+1) * SWTImage.cols + col - 1), g);
                    }
                }
            }
            ptr++;
        }
    }

    vector<int> c(num_vertices);
    int num_comp = connected_components(g, &c[0]);

	// Fill the components vector with a list of letters
	// Each element is a list of points in that letter)
    components.reserve(num_comp);
    cout << "Before filtering, " << num_comp << " components and " << num_vertices << " vertices" << endl;
    for (int j = 0; j < num_comp; j++) {
        vector<DetectText::SWTPoint2d> tmp;
        components.push_back(tmp);
    }
    for (int j = 0; j < num_vertices; j++) {
        SWTPoint2d p = revmap[j];
        (components[c[j]]).push_back(p);
    }
}


// Used to filter components vector into more accurate letters
// Limits SW value variance, aspect ratio, and size of components found earlier
void DetectText::componentStats(const vector<DetectText::SWTPoint2d> & component,
                    float & mean, float & variance, float & median,
                    int & minx, int & miny, int & maxx, int & maxy)
{
        vector<float> temp;
        temp.reserve(component.size());
        mean = 0;
        variance = 0;
        minx = 1000000;
        miny = 1000000;
        maxx = 0;
        maxy = 0;
		// Find mean SW value
		// and max and min pixel locations in both directions
        for (vector<SWTPoint2d>::const_iterator it = component.begin(); it != component.end(); it++) {
                float t = SWTImage.at<float>(it->y, it->x);
                mean += t;
                temp.push_back(t);
                miny = std::min(miny,it->y);
                minx = std::min(minx,it->x);
                maxy = std::max(maxy,it->y);
                maxx = std::max(maxx,it->x);
        }
        mean = mean / ((float)component.size());
        for (vector<float>::const_iterator it = temp.begin(); it != temp.end(); it++) {
            variance += (*it - mean) * (*it - mean);
        }
        variance = variance / ((float)component.size());
        std::sort(temp.begin(),temp.end());
        median = temp[temp.size()/2];
}



void DetectText::filterComponents() {	
	validComponents.reserve(components.size());
	compCenters.reserve(components.size());
	compMedians.reserve(components.size());
	compDimensions.reserve(components.size());
	// Bounding boxes of components
	compBB.reserve(components.size());
	for (vector<vector<SWTPoint2d> >::iterator it = components.begin(); it != components.end();it++) {
		// Computes the stroke width mean, variance, median
		float mean, variance, median;
		int minx, miny, maxx, maxy;
		componentStats((*it), mean, variance, median, minx, miny, maxx, maxy);

		// Check if variance is less than half the mean
		if (variance > 0.5 * mean) continue;

		float length = (float)(maxx-minx+1);
		float width = (float)(maxy-miny+1);

		// Check font height
		if (width > 300) continue;

		float area = length * width;
		float rminx = (float)minx;
		float rmaxx = (float)maxx;
		float rminy = (float)miny;
		float rmaxy = (float)maxy;
		// Computes the rotated bounding box
		float increment = 1./36.;
		for (float theta = increment * PI; theta<PI/2.0; theta += increment * PI) {
		    float xmin,xmax,ymin,ymax,xtemp,ytemp,ltemp,wtemp;
	        xmin = 1000000;
	        ymin = 1000000;
	        xmax = 0;
	        ymax = 0;
		    for (unsigned int i = 0; i < (*it).size(); i++) {
		        xtemp = (*it)[i].x * cos(theta) + (*it)[i].y * -sin(theta);
		        ytemp = (*it)[i].x * sin(theta) + (*it)[i].y * cos(theta);
		        xmin = std::min(xtemp,xmin);
		        xmax = std::max(xtemp,xmax);
		        ymin = std::min(ytemp,ymin);
		        ymax = std::max(ytemp,ymax);
		    }
		    ltemp = xmax - xmin + 1;
		    wtemp = ymax - ymin + 1;
		    if (ltemp*wtemp < area) {
		        area = ltemp*wtemp;
		        length = ltemp;
		        width = wtemp;
		    }
		}
		// Checks if the aspect ratio is between 1/10 and 10
		if (length/width < 1./10. || length/width > 10.) continue;

		// Creates graph representing components
		// num_nodes = number of pixels in the components
		const int num_nodes = it->size();
		Point2dFloat center;
		center.x = ((float)(maxx+minx))/2.0;
		center.y = ((float)(maxy+miny))/2.0;

		SWTPoint2d dimensions;
		dimensions.x = maxx - minx + 1;
		dimensions.y = maxy - miny + 1;

		// Upper left point of bounding box
		SWTPoint2d bb1;
		bb1.x = minx;
		bb1.y = miny;

		// Lower right point of bounding box
		SWTPoint2d bb2;
		bb2.x = maxx;
		bb2.y = maxy;
		SWTPointPair2d pair(bb1,bb2);

		// A list holding the bounding boxes of each component
		compBB.push_back(pair);
		// Stores width and height of each component 
		compDimensions.push_back(dimensions);
		// Stores median SW value for each component
		compMedians.push_back(median);
		// Stores center location for each component
		compCenters.push_back(center);
		// Stores all the components that have made it to this point (not filtered out)
		validComponents.push_back(*it);
	}
	vector<vector<SWTPoint2d > > tempComp;
	vector<SWTPoint2d > tempDim;
	vector<float > tempMed;
	vector<Point2dFloat > tempCenters;
	vector<SWTPointPair2d > tempBB;
	tempComp.reserve(validComponents.size());
	tempCenters.reserve(validComponents.size());
	tempDim.reserve(validComponents.size());
	tempMed.reserve(validComponents.size());
	tempBB.reserve(validComponents.size());

	// For every pair of components
	for (unsigned int i = 0; i < validComponents.size(); i++) {
		int count = 0;
		for (unsigned int j = 0; j < validComponents.size(); j++) {
		    if (i != j) {
				// If the center of component j lies within the bounding box of component i
		        if (compBB[i].first.x <= compCenters[j].x && compBB[i].second.x >= compCenters[j].x &&
		            compBB[i].first.y <= compCenters[j].y && compBB[i].second.y >= compCenters[j].y) {
		            count++;
		        }
		    }
		}
		// If a component has less than two other components' centers within its bounding box,
		// keep it for further processing
		if (count < 2) {
		    tempComp.push_back(validComponents[i]);
		    tempCenters.push_back(compCenters[i]);
		    tempMed.push_back(compMedians[i]);
		    tempDim.push_back(compDimensions[i]);
		    tempBB.push_back(compBB[i]);
		}
	}
	validComponents = tempComp;
	compDimensions = tempDim;
	compMedians = tempMed;
	compCenters = tempCenters;
	compBB = tempBB;

	compDimensions.reserve(tempComp.size());
	compMedians.reserve(tempComp.size());
	compCenters.reserve(tempComp.size());
	validComponents.reserve(tempComp.size());
	compBB.reserve(tempComp.size());

	cout << "After filtering " << validComponents.size() << " components" << endl;
}


// Renders the components on original image
// Don't need this in the final program
void DetectText::renderComponentsWithBoxes () {
	Mat output = componentDisplay.clone();    

    bb.reserve(compBB.size());
    for (auto& it : compBB) {
        Point2i p0 = cvPoint(it.first.x,  it.first.y);
        Point2i p1 = cvPoint(it.second.x, it.second.y);
        SWTPointPair2i pair(p0, p1);
        bb.push_back(pair);
    }

    int count = 0;
    for (auto it : bb) {
        Scalar c;
        if (count % 3 == 0) {
            c = BLUE;
        }
        else if (count % 3 == 1) {
            c = GREEN;
        }
        else {
            c = RED;
        }
        count++;
        rectangle(output, it.first, it.second, c, 1);
    }

    imwrite ("components.png", output);
}


// Determines whether two chains share and endpoint
bool DetectText::sharesOneEnd( Chain c0, Chain c1) {
    if (c0.p == c1.p || c0.p == c1.q || c0.q == c1.q || c0.q == c1.p) {
        return true;
    }
    else {
        return false;
    }
}

// Sorts changes based on 
bool DetectText::chainSortDist (const Chain &lhs, const Chain &rhs) {
    return lhs.dist < rhs.dist;
}

bool DetectText::chainSortLength (const Chain &lhs, const Chain &rhs) {
    return lhs.components.size() > rhs.components.size();
}


void DetectText::makeChains() {
	chains.clear();
    assert (compCenters.size() == validComponents.size());
    // For each valid component, 
    vector<Point3dFloat> colorAverages;
    colorAverages.reserve(validComponents.size());
    for (vector<vector<DetectText::SWTPoint2d> >::iterator it = validComponents.begin(); it != validComponents.end();it++) {
        Point3dFloat mean;
        mean.x = 0;
        mean.y = 0;
        mean.z = 0;
        int num_points = 0;
        for (vector<DetectText::SWTPoint2d>::iterator pit = it->begin(); pit != it->end(); pit++) {
            mean.x += (float) input.at<uchar>(pit->y, (pit->x)*3 );
            mean.y += (float) input.at<uchar>(pit->y, (pit->x)*3+1 );
            mean.z += (float) input.at<uchar>(pit->y, (pit->x)*3+2 );
            num_points++;
        }
        mean.x = mean.x / ((float)num_points);
        mean.y = mean.y / ((float)num_points);
        mean.z = mean.z / ((float)num_points);
        colorAverages.push_back(mean);
    }

    // Form all eligible pairs and calculate the direction of each
    for ( unsigned int i = 0; i < validComponents.size(); i++ ) {
        for ( unsigned int j = i + 1; j < validComponents.size(); j++ ) {

			// Check that SW median value for both components are similar
			// and that the components are roughly the same size
            if ( (compMedians[i]/compMedians[j] <= 2.0 || compMedians[j]/compMedians[i] <= 2.0) &&
                 (compDimensions[i].y/compDimensions[j].y <= 2.0 || compDimensions[j].y/compDimensions[i].y <= 2.0)) {

				// Distance between the two centers
                float dist = (compCenters[i].x - compCenters[j].x) * (compCenters[i].x - compCenters[j].x) +
                             (compCenters[i].y - compCenters[j].y) * (compCenters[i].y - compCenters[j].y);

				// "Distance" metric between average component colors
                float colorDist = (colorAverages[i].x - colorAverages[j].x) * (colorAverages[i].x - colorAverages[j].x) +
                                  (colorAverages[i].y - colorAverages[j].y) * (colorAverages[i].y - colorAverages[j].y) +
                                  (colorAverages[i].z - colorAverages[j].z) * (colorAverages[i].z - colorAverages[j].z);

				// If distance between two components is less than 3 times the width of the larger component
				// TODO: This actually uses the height IF the height is smaller than the width; is that necessary?
                if (dist < 9*(float)(std::max(std::min(compDimensions[i].x,compDimensions[i].y),std::min(compDimensions[j].x,compDimensions[j].y)))
                    *(float)(std::max(std::min(compDimensions[i].x,compDimensions[i].y),std::min(compDimensions[j].x,compDimensions[j].y)))
                    && colorDist < 1600) {
					
					// Create chain to hold the valid pair
                    Chain c;
                    c.p = i;						// The ith components ID
                    c.q = j;						// The jth components ID
                    vector<int> comps;				// Currently just stores a list of the two IDs
                    comps.push_back(c.p);			// (will add middle components in next step)
                    comps.push_back(c.q);
                    c.components = comps;
                    c.dist = dist;					// Stores distance between chain's endpoints
					// Stores direction of chain by finding vector between the two components' centers
                    float d_x = (compCenters[i].x - compCenters[j].x);
                    float d_y = (compCenters[i].y - compCenters[j].y);
                    float mag = sqrt(d_x*d_x + d_y*d_y);
                    d_x = d_x / mag;
                    d_y = d_y / mag;
                    DetectText::Point2dFloat dir;
                    dir.x = d_x;
                    dir.y = d_y;
                    c.direction = dir;
                    chains.push_back(c);
                }
            }
        }
    }
    cout << chains.size() << " eligible pairs" << endl;

	// Sorts pairs from least to greatest distance between components
    std::sort(chains.begin(), chains.end(), &DetectText::chainSortDist);

	cerr << endl;
	// Max angle between two pair directions to consider them part of the same line
    const float strictness = PI/6.0;

    // Merge chains
	// This loop runs until an iteration has elapsed without any merging
	// At this point, all pairs will have been chained together as much as possible
    int merges = 1;
    while (merges > 0) {
        for (unsigned int i = 0; i < chains.size(); i++)
            chains[i].merged = false;
        merges = 0;
        vector<DetectText::Chain> newchains;
		// For every pair of pairs
        for (unsigned int i = 0; i < chains.size(); i++) {
            for (unsigned int j = 0; j < chains.size(); j++) {
                if (i != j) {
					// If neither pair has been merged at this point, and if they share and endpoint
                    if (!chains[i].merged && !chains[j].merged && sharesOneEnd(chains[i],chains[j])) {
						// If the chains share the first point
                        if (chains[i].p == chains[j].p) {
							// If the two chains have roughly the same direction
                            if (acos(chains[i].direction.x* -chains[j].direction.x + chains[i].direction.y * -chains[j].direction.y) < strictness) {
                                // Change endpoint to include the second chain's
								chains[i].p = chains[j].q;
								// Add all the components from chain j to chain i
                                for (vector<int>::iterator it = chains[j].components.begin(); it != chains[j].components.end(); it++) {
                                    chains[i].components.push_back(*it);
                                }
								// Update the chain's direction
                                float d_x = (compCenters[chains[i].p].x - compCenters[chains[i].q].x);
                                float d_y = (compCenters[chains[i].p].y - compCenters[chains[i].q].y);
                                chains[i].dist = d_x * d_x + d_y * d_y;

                                float mag = sqrt(d_x*d_x + d_y*d_y);
                                d_x = d_x / mag;
                                d_y = d_y / mag;
                                DetectText::Point2dFloat dir;
                                dir.x = d_x;
                                dir.y = d_y;
                                chains[i].direction = dir;

								// Prevent chain j from being accessed again
                                chains[j].merged = true;

								// Keep looping through, since more chains have been merged
                                merges++;
                            }
						// The other blocks follow a similar process
                        } else if (chains[i].p == chains[j].q) {
                            if (acos(chains[i].direction.x * chains[j].direction.x + chains[i].direction.y * chains[j].direction.y) < strictness) {
                                chains[i].p = chains[j].p;
                                for (vector<int>::iterator it = chains[j].components.begin(); it != chains[j].components.end(); it++) {
                                    chains[i].components.push_back(*it);
                                }
                                float d_x = (compCenters[chains[i].p].x - compCenters[chains[i].q].x);
                                float d_y = (compCenters[chains[i].p].y - compCenters[chains[i].q].y);
                                float mag = sqrt(d_x*d_x + d_y*d_y);
                                chains[i].dist = d_x * d_x + d_y * d_y;

                                d_x = d_x / mag;
                                d_y = d_y / mag;

                                DetectText::Point2dFloat dir;
                                dir.x = d_x;
                                dir.y = d_y;
                                chains[i].direction = dir;
                                chains[j].merged = true;
                                merges++;
                            }
                        } else if (chains[i].q == chains[j].p) {
                            if (acos(chains[i].direction.x * chains[j].direction.x + chains[i].direction.y * chains[j].direction.y) < strictness) {
                                chains[i].q = chains[j].q;
                                for (vector<int>::iterator it = chains[j].components.begin(); it != chains[j].components.end(); it++) {
                                    chains[i].components.push_back(*it);
                                }
                                float d_x = (compCenters[chains[i].p].x - compCenters[chains[i].q].x);
                                float d_y = (compCenters[chains[i].p].y - compCenters[chains[i].q].y);
                                float mag = sqrt(d_x*d_x + d_y*d_y);
                                chains[i].dist = d_x * d_x + d_y * d_y;


                                d_x = d_x / mag;
                                d_y = d_y / mag;
                                DetectText::Point2dFloat dir;
                                dir.x = d_x;
                                dir.y = d_y;

                                chains[i].direction = dir;
                                chains[j].merged = true;
                                merges++;
                            }
						// If the chains share their second point
                        } else if (chains[i].q == chains[j].q) {
                            if (acos(chains[i].direction.x* -chains[j].direction.x + chains[i].direction.y * -chains[j].direction.y) < strictness) {
                                chains[i].q = chains[j].p;
                                for (vector<int>::iterator it = chains[j].components.begin(); it != chains[j].components.end(); it++) {
                                    chains[i].components.push_back(*it);
                                }
                                float d_x = (compCenters[chains[i].p].x - compCenters[chains[i].q].x);
                                float d_y = (compCenters[chains[i].p].y - compCenters[chains[i].q].y);
                                chains[i].dist = d_x * d_x + d_y * d_y;

                                float mag = sqrt(d_x*d_x + d_y*d_y);
                                d_x = d_x / mag;
                                d_y = d_y / mag;
                                DetectText::Point2dFloat dir;
                                dir.x = d_x;
                                dir.y = d_y;
                                chains[i].direction = dir;
                                chains[j].merged = true;
                                merges++;
                            }
                        }
                    }
                }
            }
        }

		// At each iterations, discard the pairs that have been grouped into a larger chain
		// and sort the remaining chains by length

		// TODO: Could this be put off until after the while loop finishes? 
		// Seems unnecessary to do this AND have the if(!chain[i].merged) check
        for (unsigned int i = 0; i < chains.size(); i++) {
            if (!chains[i].merged) {
                newchains.push_back(chains[i]);
            }
        }
        chains = newchains;
        std::stable_sort(chains.begin(), chains.end(), &DetectText::chainSortLength);
    }

	
    vector<DetectText::Chain> newchains;
    newchains.reserve(chains.size());
	// Keep only the chains that have at least 3 components
    for (vector<Chain>::iterator cit = chains.begin(); cit != chains.end(); cit++) {
        if (cit->components.size() >= 3) {
            newchains.push_back(*cit);
        }
    }
    chains = newchains;
    cout << chains.size() << " chains after merging" << endl;

	// Renders the chained words
	// Probably not necessary for the final program
    int count = 0;
    for (auto it : chains) {
        Scalar c;
        if (count % 3 == 0) {
            c = BLUE;
        }
        else if (count % 3 == 1) {
            c = GREEN;
        }
        else {
            c = RED;
        }
        count++;

		for (int i : it.components) {
        	rectangle(chainDisplay, bb[i].first, bb[i].second, c, 1);
		}
    }
    imwrite ("chains.png", chainDisplay);
}


// Public function
// Runs the whole process, from preprocessing to word chaining
void DetectText::detectText() {
	// Runs preprocessing and stroke width transform
	edgeAndGradient();
	strokeWidthTransform();
	SWTMedianFilter();
	displaySWT();

	// Calculate legally connect components from SWT and gradient image.
	findLegallyConnectedComponents();
	filterComponents();
	renderComponentsWithBoxes();
	
	// Chaining together valid words
	makeChains();
}
