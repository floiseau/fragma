SetFactory("OpenCASCADE");

// Create a cube
Box(1) = {0, 0, 0, 1, 1, 1};

// Add physical groups
Physical Volume("domain", 15) = {1};
Physical Surface("bot", 13) = {3};
Physical Surface("top", 14) = {4};
