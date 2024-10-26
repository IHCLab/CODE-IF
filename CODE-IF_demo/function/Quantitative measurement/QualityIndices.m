function Out = QualityIndices(I_HS,I_REF,ratio, max, min)
%--------------------------------------------------------------------------
% Quality Indices
%
% USAGE
%   Out = QualityIndices(I_HS,I_REF,ratio)
%
% INPUT
%   I_HS  : target HS data (rows,cols,bands)
%   I_REF : reference HS data (rows,cols,bands)
%   ratio : GSD ratio between HS and MS imagers
%
% OUTPUT
%   Out.psnr : PSNR
%   Out.sam  : SAM
%   Out.rmse : RMSE
%   Out.ergas: ERGAS
%--------------------------------------------------------------------------
    psnr = PSNR(I_HS,I_REF);
    Out.psnr = psnr.ave;
    [angle_SAM,~] = SAM(I_HS,I_REF);
    Out.sam = angle_SAM;
    Out.rmse = RMSE(I_HS,I_REF,max,min);
    Out.ergas = ERGAS(I_HS,I_REF,ratio); 

end 
