%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% script by Giacomo Handjaras, Francesca Setti %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function r=fast_corr(X,Y)
X = bsxfun(@minus,X,nansum(X,1)./size(X,1));
Y = bsxfun(@minus,Y,nansum(Y,1)./size(Y,1));
X = X.*repmat(sqrt(1./max(eps,nansum(abs(X).^2,1))),[size(X,1),1]);
Y = Y.*repmat(sqrt(1./max(eps,nansum(abs(Y).^2,1))),[size(Y,1),1]);
r = nansum(X.*Y);
end
