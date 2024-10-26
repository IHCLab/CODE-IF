function out = PSNR(tar,ref)
%--------------------------------------------------------------------------
% Peak signal to noise ratio (PSNR)
%
% USAGE
%   out = PSNR(ref,tar,mask)
%
% INPUT
%   ref : reference HS data (rows,cols,bands)
%   tar : target HS data (rows,cols,bands)
%   mask: binary mask (rows,cols) (optional)
%
% OUTPUT
%   out : PSNR (scalar)
%
%--------------------------------------------------------------------------
[~,~,bands] = size(ref);

ref = reshape(ref,[],bands);
tar = reshape(tar,[],bands);
msr = mean((ref-tar).^2,1);
max2 = max(tar,[],1).^2;
    
psnrall = 10*log10(max2./msr);
out.all = psnrall;
out.ave = mean(psnrall);