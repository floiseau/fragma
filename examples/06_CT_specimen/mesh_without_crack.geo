SetFactory("OpenCASCADE");

//// Parameters ////////////////////////
// Numerical
lcmin = DefineNumber[ 1e-3, Name "Parameters/lcmin" ];
lcmax = DefineNumber[ 10*lcmin, Name "Parameters/lcmax" ];
// Boundaries
W =  DefineNumber[ 1.0, Name "Parameters/W" ];
// Notch
na = DefineNumber[ 30, Name "Parameters/na" ]; // Notch angle
nh = 0.02*W;        // Notch height
nw = 0.2*W;  // Notch depht (from center of pin holes)
// Pin holes
phh = 0.325*W;
D = 0.25*W;
// Crack
a0 = DefineNumber[ 0.2, Name "Parameters/a0" ];
aw = DefineNumber[ 0.05, Name "Parameters/aw" ];

//// Points ////////////////////////////
// Boundary
Point(1) = {-0.25*W, 0, 0, lcmax};
Point(2) = {      W, 0, 0, lcmax};
Point(3) = {      W, 1.25*W, 0, lcmax};
Point(4) = {-0.25*W, 1.25*W, 0, lcmax};
// Notch
Point(5) = {-0.25*W, 0.625*W+nh/2, 0, lcmax};
Point(6) = {nw-nh/2/Tan(na/2*Pi/180), 0.625*W+nh/2, 0, lcmax};
Point(7) = {nw, 0.625*W, 0, lcmax}; // Notch tip
Point(8) = {nw-nh/2/Tan(na/2*Pi/180), 0.625*W-nh/2, 0, lcmax};
Point(9) = {-0.25*W, 0.625*W-nh/2, 0, lcmax};

//// Lines /////////////////////////////
// Boundaries + Notch
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 5};
Line(5) = {5, 6};
Line(6) = {6, 7};
Line(7) = {7, 8};
Line(8) = {8, 9};
Line(9) = {9, 1};
// Pin Holes
Circle(10) = {0, phh, 0, D/2, 0, 2*Pi};
Circle(11) = {0, 1.25*W-phh, 0, D/2, 0, 2*Pi};

//// Surfaces //////////////////////////
Curve Loop(1) = {8, 9, 1, 2, 3, 4, 5, 6, 7};
Curve Loop(2) = {10};
Curve Loop(3) = {11};
Plane Surface(1) = {1, 2, 3};

//// Physical group ////////////////////
Physical Surface("Domain", 12) = {1};
Physical Curve("crack", 13) = {7, 6};
Physical Curve("bot_pin", 14) = {10};
Physical Curve("top_pin", 15) = {11};

//// Element size /////////////////////
// Number of points to discretize circle
Mesh.MinimumCirclePoints = (Pi*D)/lcmax;
// Create geometric entities
Point(12) = {D/2, 1.25*W/2, 0, lcmax};
Point(13) = {W, 1.25*W/2, 0, lcmax};
Line(12) = {12, 13};
// Distance fields
Field[1] = Distance;
Field[1].CurvesList = {12};
Field[1].Sampling = 100;
// Threshold field
Field[2] = Threshold;
Field[2].InField = 1;
Field[2].SizeMin = lcmin;
Field[2].SizeMax = lcmax;
Field[2].DistMin = 2*nh;
Field[2].DistMax = 4*nh;
// Apply field 2 as element size
Background Field = 2;

