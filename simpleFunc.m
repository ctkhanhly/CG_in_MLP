[X,Y] = meshgrid(1:0.5:10,1:20);
Z = exp(-X.*Y);
surf(X,Y,Z);