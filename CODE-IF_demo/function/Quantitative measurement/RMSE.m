function out = RMSE(tar,ref,max, min)
%--------------------------------------------------------------------------
% Root mean squared error (RMSE)
%
% USAGE
%   out = RMSE(ref,tar)
%
% INPUT
%   ref : reference HS data (rows,cols,bands)
%   tar : target HS data (rows,cols,bands)
%
% OUTPUT
%   out : RMSE (scalar)
%
%--------------------------------------------------------------------------
[rows,cols,bands] = size(ref);
ref=ref*(max-min)+min;
tar=tar*(max-min)+min;
out = (sum(sum(sum((tar-ref).^2)))/(rows*cols*bands)).^0.5;
% out=out*max;
% out = out*(max-min)+min;