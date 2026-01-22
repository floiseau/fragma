//// Parameters
// Mesh refinement
DefineConstant[ N = {100, Name "Parameters/N"}];
DefineConstant[ alpha = {0, Name "Parameters/alpha"}];

// Geometry
L = 1.0;
// Half crack length
a = 0.05*L;
// Crack orientation
alphaRad = alpha * Pi/180;
// Numerical
h = L/N;
h_crack = h/100;

//// Points
// Bot
Point(11) = {-L/2, -L/2, 0, h};
Point(12) = { L/2, -L/2, 0, h};
Point(13) = { L/2, 0, 0, h};			      	// Mid right node
Point(14) = { a*Cos(alphaRad), a*Sin(alphaRad), 0, h_crack}; 	// Left crack tip
Point(15) = {-a*Cos(alphaRad),-a*Sin(alphaRad), 0, h_crack}; 	// Right crack tip
Point(16) = {-L/2, 0, 0, h};     			// Mid left node
// Top
Point(21) = {-L/2, L/2, 0, h};
Point(22) = { L/2, L/2, 0, h};

//// Lines
// Bot
Line(11) = {11, 12};
Line(12) = {12, 13};
Line(13) = {13, 14};
Line(14) = {14, 15}; // Bot crack line
Line(15) = {15, 16};
Line(16) = {16, 11};
// // Top
Line(21) = {21, 22};
Line(22) = {22, 13};
// // Line(13)
Line(24) = {14, 15}; // Top crack line
// Line(15);
Line(26) = {16, 21};


//// Surfaces
// Bot
Curve Loop(1) = {11, 12, 13, 14, 15, 16};
Plane Surface(1) = {1};
// Top
Curve Loop(2) = {21, 22, 13, 24, 15, 26};
Plane Surface(2) = {2};

//// Physical groups
// Domain
Physical Surface("domain", 21) = {1, 2};
// Boundaries
Physical Curve("bot", 11) = {11};
Physical Curve("top", 12) = {21};
// Crack
Physical Curve("crack", 13) = {14, 24};
