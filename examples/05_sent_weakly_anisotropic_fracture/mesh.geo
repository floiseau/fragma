//// Parameters
L = DefineNumber[ 1, Name "Parameters/L" ];
h = DefineNumber[ 0.01, Name "Parameters/h" ];

//// Points
// Bot
Point(11) = {0, 0, 0, h};
Point(13) = {L, 0, 0, h};
// Mid (bot)
Point(21) = {0, L/2-h/2, 0, h};
Point(22) = {L/2, L/2-h/2, 0, h};
// Mid (top)
Point(31) = {0, L/2+h/2, 0, h};
Point(32) = {L/2, L/2+h/2, 0, h};
// Top
Point(41) = {0, L, 0, h};
Point(43) = {L, L, 0, h};

//// Lines
Line(1) = {11, 13};
Line(2) = {13, 43};
Line(3) = {43, 41};
Line(4) = {41, 31};
Line(5) = {31, 32};
Line(6) = {32, 22};
Line(7) = {22, 21};
Line(8) = {21, 11};

//// Surfaces
// Bottom part
Curve Loop(1) = {1, 2, 3, 4, 5, 6, 7, 8};
Plane Surface(1) = {1};

//// Physical groups
// Domain
Physical Surface("domain", 1) = {1};
// Lines
Physical Curve("bot", 2) = {1};
Physical Curve("top", 3) = {3};
Physical Curve("crack", 10) = {5, 6, 7};
