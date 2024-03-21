//// Parameters
// Geometry
L = 1;
// Mechanical
ell = 0.02;
// Numerical
h = ell;
h_min = ell/5;

//// Points
// Bot
Point(11) = {0, 0, 0, h};
Point(12) = {L, 0, 0, h};
Point(13) = {L, L, 0, h};
Point(14) = {0, L, 0, h};

//// Lines
// Bot
Line(11) = {11, 12};
Line(12) = {12, 13};
Line(13) = {13, 14};
Line(14) = {14, 11};

//// Surfaces
// Bot
Curve Loop(1) = {11, 12, 13, 14};
Plane Surface(1) = {1};

//// Physical groups
// Domain
Physical Surface("domain", 21) = {1};
// Boundaries
Physical Curve("bot", 11) = {11};
Physical Curve("top", 12) = {13};

//// Element size
// Define the crack line
Point(21) = {0, L/2, 0, h};
Point(22) = {L, L/2, 0, h};
Line(21) = {21, 22};
// Refine around the crack line
Field[1] = Distance;
Field[1].CurvesList = {21};
Field[1].Sampling = 1000;
Field[2] = Threshold;
Field[2].InField = 1;
Field[2].DistMin = 2*ell;
Field[2].DistMax = 5*ell;
Field[2].SizeMin = h_min;
Field[2].SizeMax = h;
// Set the background field
Background Field = 2;
