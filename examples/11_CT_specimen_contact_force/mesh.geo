//// Options
SetFactory("OpenCASCADE");

//// Parameters
// Numeric
lcmin = DefineNumber[ 0.15e-3, Name "Parameters/lcmax" ];
lcmax = 4*lcmin;
// Geometry
W = DefineNumber[   40e-3, Name "Parameters/W" ];
H = DefineNumber[ 38.4e-3, Name "Parameters/H" ];
// Extenso
alpha = DefineNumber[ 60, Name "Parameters/alpha" ];
beta  = DefineNumber[ 90, Name "Parameters/beta" ];
w_ext = DefineNumber[ 1e-3, Name "Parameters/w_ext" ];
h_ext = DefineNumber[ 5e-3, Name "Parameters/h_ext" ];
// Notch
n_h = DefineNumber[ 2e-3, Name "Parameters/n_h" ];
n_w = DefineNumber[ 20e-3, Name "Parameters/n_w" ];
n_ang = DefineNumber[ 60, Name "Parameters/n_ang" ];
// Pin holes
p_w = DefineNumber[ 8e-3, Name "Parameters/p_w" ];
p_h = DefineNumber[ 10.4e-3, Name "Parameters/p_h" ];
p_r = DefineNumber[ 4e-3, Name "Parameters/p_r" ];
// Pre-crack
a_0 = DefineNumber[ 2.5e-3, Name "Parameters/a_0" ];
// Pre-crack
// Convert angle to radians
al = alpha * Pi/180;
be = beta  * Pi/180;
na = n_ang * Pi/180;

//// Points
// Bot
Point(101) = {0, 0, 0, lcmax};
Point(102) = {W, 0, 0, lcmax};
Point(103) = {W, H/2, 0, lcmax};       // Mid right
Point(104) = {n_w+a_0, H/2, 0, lcmax}; // Crack tip
Point(105) = {n_w, H/2, 0, lcmax};     // Bot crack lip
Point(106) = {n_w-n_h/2/Tan(na/2), H/2-n_h/2, 0, lcmax};
Point(107) = {w_ext+1/2*(h_ext-n_h)/Tan(al), H/2-n_h/2, 0, lcmax};
Point(108) = {w_ext, H/2-h_ext/2, 0, lcmax};
Point(109) = {0, H/2-h_ext/2+w_ext/Tan(al), 0, lcmax};
// Top
Point(201) = {0, H, 0, lcmax};
Point(202) = {W, H, 0, lcmax};
// Point(103)
// Point(104) (crack tip)
Point(205) = {n_w, H/2, 0, lcmax};     // Top crack lip
Point(206) = {n_w-n_h/2/Tan(na/2), H/2+n_h/2, 0, lcmax};
Point(207) = {w_ext+1/2*(h_ext-n_h)/Tan(al), H/2+n_h/2, 0, lcmax};
Point(208) = {w_ext, H/2+h_ext/2, 0, lcmax};
Point(209) = {0, H/2+h_ext/2-w_ext/Tan(al), 0, lcmax};
// Bot pin hole
Point(301) = {p_w, p_h, 0, lcmax};
Point(302) = {p_w+p_r, p_h, 0, lcmax};
Point(303) = {p_w-p_r, p_h, 0, lcmax};
// Top pin hole
Point(401) = {p_w, H-p_h, 0, lcmax};
Point(402) = {p_w+p_r, H-p_h, 0, lcmax};
Point(403) = {p_w-p_r, H-p_h, 0, lcmax};

//// Lines
// Bot
Line(101) = {101, 102};
Line(102) = {102, 103};
Line(103) = {103, 104};
Line(104) = {104, 105};
Line(105) = {105, 106};
Line(106) = {106, 107};
Line(107) = {107, 108};
Line(108) = {108, 109};
Line(109) = {109, 101};
// Top
Line(201) = {201, 202};
Line(202) = {202, 103};
// Line(103)
Line(204) = {104, 205};
Line(205) = {205, 206};
Line(206) = {206, 207};
Line(207) = {207, 208};
Line(208) = {208, 209};
Line(209) = {209, 201};
// Bot pin hole
Circle(301) = {303, 301, 302};
Circle(302) = {302, 301, 303};
// Top pin hole
Circle(401) = {403, 401, 402};
Circle(402) = {402, 401, 403};

//// Surfaces
// Bot
Curve Loop(101) = {101, 102, 103, 104, 105, 106, 107, 108, 109};
Curve Loop(102) = {301, 302};
Plane Surface(101) = {101, 102};
// Pot
Curve Loop(201) = {201, 202, 103, 204, 205, 206, 207, 208, 209};
Curve Loop(202) = {401, 402};
Plane Surface(201) = {201, 202};

//// Physical groups
// Domain
Physical Surface("domain", 1) = {101, 201};
// Pins
Physical Curve("bot_pin", 2) = {302};
Physical Curve("top_pin", 3) = {401};
// Crack
Physical Curve("crack", 4) = {104, 204};

//// Element size field
// Distance fields
Field[1] = Distance;
Field[1].CurvesList = {103, 104, 204};
Field[1].Sampling = 100;
// Threshold field
Field[2] = Threshold;
Field[2].InField = 1;
Field[2].SizeMin = lcmin;
Field[2].SizeMax = lcmax;
Field[2].DistMin = 0.05*H;
Field[2].DistMax = 0.3*H;
// Apply field 2 as element size
Background Field = 2;


