// Parameters
Geometry.AutoCoherence = 0;

//// Parameters
lc = DefineNumber[ 0.01, Name "Parameters/lc" ];
L = DefineNumber[ 1, Name "Parameters/L" ];

//// Points
// Bot
Point(11) = {0, 0, 0, lc};
Point(12) = {L, 0, 0, lc};
// Mid
Point(21) = {L, L/2, 0, lc};
Point(22) = {L/2, L/2, 0, lc}; // Crack tip
Point(23) = {0, L/2, 0, lc}; // Bot crack lip
Point(24) = {0, L/2, 0, lc}; // Top crack lip
// Top
Point(31) = {0, L, 0, lc};
Point(32) = {L, L, 0, lc};

//// Lines
// Bottom part
Line(11) = {11, 12};
Line(12) = {12, 21};
Line(13) = {21, 22};
Line(14) = {22, 23};
Line(15) = {23, 11};
// Top part
Line(31) = {31, 32};
Line(32) = {32, 21};
// Line 13 (right to crack tip)
Line(34) = {22, 24};
Line(35) = {24, 31};

//// Surfaces
// Bottom part
Curve Loop(1) = {11, 12, 13, 14, 15};
Plane Surface(1) = {1};
// // Top part
Curve Loop(2) = {31, 32, 13, 34, 35};
Plane Surface(2) = {2};

//// Physical groups
// Domain
Physical Surface("domain", 36) = {1, 2};
// Lines
Physical Curve("bot", 37) = {11};
Physical Curve("top", 38) = {31};
Physical Curve("crack", 39) = {14, 34};
