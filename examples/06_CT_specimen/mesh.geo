//// Notes /////////////////////////////
// The pre-crack in this methc is a kind
// of notch with a fixed thickness.
// See the meshes from examples 07 or 05
// to make a infinitely thin pre-crack.
////////////////////////////////////////

//// Options ///////////////////////////
SetFactory("OpenCASCADE");

//// Parameters ////////////////////////
// Numerical
lcmin = DefineNumber[ 2e-3, Name "Parameters/lcmin" ];
lcmax = DefineNumber[ 10*lcmin, Name "Parameters/lcmax" ];
// Boundaries
W =  DefineNumber[ 1.0, Name "Parameters/W" ];
H =  1.2*W;
// Notch
na = DefineNumber[ 30, Name "Parameters/na" ]; // Notch angle
nh = 0.02*W;                                   // Notch height
nw = 0.2*W;                             // Notch depht (from center of pin holes)
// Pin holes
phh = 0.325*W;
D = 0.25*W;
// Crack
a0 = DefineNumber[ 0.1*W, Name "Parameters/a0" ];
aw = DefineNumber[ 1e-6, Name "Parameters/aw" ];

//// Points ////////////////////////////
// Boundary
Point(1) = {-0.25*W, 0, 0, lcmax};
Point(2) = {      W, 0, 0, lcmax};
Point(3) = {      W, H, 0, lcmax};
Point(4) = {-0.25*W, H, 0, lcmax};
// Notch + Crack
Point(5) = {-0.25*W, H/2+nh/2, 0, lcmax};
Point(6) = {nw-nh/2/Tan(na/2*Pi/180), H/2+nh/2, 0, lcmax};
Point(7) = {nw   , H/2+aw/2, 0, lcmax}; // Notch tip top
Point(8) = {nw+a0, H/2     , 0, lcmax}; // Crack tip
Point(9) = {nw   , H/2-aw/2, 0, lcmax}; // Notch tip bot
Point(10) = {nw-nh/2/Tan(na/2*Pi/180), H/2-nh/2, 0, lcmax};
Point(11) = {-0.25*W, H/2-nh/2, 0, lcmax};

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
Line(9) = {9, 10};
Line(10) = {10, 11};
Line(11) = {11, 1};
// Pin Holes
Circle(12) = {0, phh, 0, D/2, 0, 2*Pi};
Circle(13) = {0, H-phh, 0, D/2, 0, 2*Pi};

//// Surfaces //////////////////////////
Curve Loop(1) = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
Curve Loop(2) = {12};
Curve Loop(3) = {13};
Plane Surface(1) = {1, 2, 3};

//// Physical group ////////////////////
Physical Surface("Domain", 20) = {1};
Physical Curve("crack", 21) = {7, 8};
Physical Curve("bot_pin", 22) = {12};
Physical Curve("top_pin", 23) = {13};

//// Element size /////////////////////
// Number of points to discretize circle
Mesh.MinimumCirclePoints = (Pi*D)/lcmax;
// Create geometric entities
Point(20) = {D/2, H/2, 0, lcmax};
Point(21) = {W, H/2, 0, lcmax};
Line(20) = {20, 21};
// Distance fields
Field[1] = Distance;
Field[1].CurvesList = {20};
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
