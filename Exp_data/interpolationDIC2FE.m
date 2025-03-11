% DESCRIPTION
%  Script to perform the interpolation of digital image correlation data to
%  finite element node positions for constitutive model calibration
%
%  One .csv-file with the measured displacement data has to be provided
%  along with a file containing the coordinates of the finite element
%  nodes. It has to be assured that both coordinates from DIC and FE are
%  aligned with each other
%
% CREATOR
%  Jendrik-Alexander Tröger, Institute of Applied Mechanics, Clausthal
%  University of Technology

clc
clear
close all

% read DIC-data -- x-coord. [mm]  y-coord. [mm]  x-disp. [mm]   y-disp. [mm]
DICdata = readmatrix("20231116_displacements_raw.csv");

% read finite element node positions -- node id  x-coord. [mm]  y-coord. [mm]
coordinatesFE = readmatrix("20231116_fem_node_coordinate.csv");

% build spatial interpolations for axial and lateral displacements
F_ux = scatteredInterpolant(DICdata(:,1),DICdata(:,2),DICdata(:,3));
F_uy = scatteredInterpolant(DICdata(:,1),DICdata(:,2),DICdata(:,4));
% set interpolation method to linear (default)
F_ux.Method = 'linear';
F_uy.Method = 'linear';

% consider only finite element nodes within domain x = [0 80], y = [0 20]
idx = (1:length(coordinatesFE))';
logIdx = idx(coordinatesFE(:,2) >= 0);

% evaluate interpolants at finite element node positions within domain 
% x = [0 80], y = [0 20] 
FEdisp(:,1) = F_ux(coordinatesFE(logIdx,2),coordinatesFE(logIdx,3));
FEdisp(:,2) = F_uy(coordinatesFE(logIdx,2),coordinatesFE(logIdx,3));