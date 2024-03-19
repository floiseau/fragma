//// Parameters
// Geometry
L = 1e-3;
// Crack tip position
a = L/4;
// Numerical
h = L/100;

//// Points
// Bot
Point(11) = {0, 0, 0, h};
Point(12) = {L, 0, 0, h};
Point(13) = {L, L/2, 0, h};      // Mid right node
Point(14) = {a, L/2, 0, h};    // Crack tip
Point(15) = {0, L/2, 0, h};      // Bot crack lip
// Top
Point(21) = {0, L, 0, h};
Point(22) = {L, L, 0, h};
// Point(13) // Mid right node
// Point(14) // Crack tip
Point(25) = {0, L/2, 0, h};      // Top crack lip

//// Lines
// Bot
Line(11) = {11, 12};
Line(12) = {12, 13};
Line(13) = {13, 14};
Line(14) = {14, 15};
Line(15) = {15, 11};
// Top
Line(21) = {21, 22};
Line(22) = {22, 13};
// Line(13)
Line(24) = {14, 25};
Line(25) = {25, 21};

//// Surfaces
// Bot
Curve Loop(1) = {11, 12, 13, 14, 15};
Plane Surface(1) = {1};
// Top
Curve Loop(2) = {21, 22, 13, 24, 25};
Plane Surface(2) = {2};

//// Physical groups
// Domain
Physical Surface("domain", 21) = {1, 2};
// Boundaries
Physical Curve("bot", 11) = {11};
Physical Curve("top", 12) = {21};
// Crack
Physical Curve("crack", 13) = {14, 24};
