//// Parameters
// Geometry
L = 1;
// Mechanical
ell = 0.05;
// Numerical
Nnodes = 100;
h = L/Nnodes;

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

//// Mesh
// Transfinite entities
Transfinite Curve {11, 13} = Nnodes Using Progression 1;
Transfinite Curve {14, 12} = Nnodes Using Progression 1;
Transfinite Surface {1};
// Recombine elements into quads
Recombine Surface {1};

//// Physical groups
// Domain
Physical Surface("domain", 21) = {1};
// Boundaries
Physical Curve("bot", 11) = {11};
Physical Curve("top", 12) = {13};
