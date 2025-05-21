function ann=trainann(ann,x,y, no_of_epochs,batch_size)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

m=1;
%only images either grayvalues or RGB
m_index=1;
if size(x,4) > 1 %RGB
    m=size(x,4);
    m_index=4; %example index
else
    m=size(x,3);
    m_index=3;
end
no_of_batches = m/batch_size; %should be integer
if rem(m, batch_size) ~=0
    error('no_of_batches should be integer');
end

if ann.loss_func == 'auto'
   ann.loss_func = 'quad'; %quadtratic
   if ann.layers{ann.no_of_layers}.act_func == 'sigm'
       ann.loss_func = 'cros' ; %cross_entropy';
   elseif ann.layers{ann.no_of_layers}.act_func == 'tanh'
       ann.loss_func = 'quad';
   
       
   end
elseif strcmp(ann.loss_func, 'cros') == 1 & strcmp(ann.layers{ann.no_of_layers}.act_func, 'sigm') == 0
    display 'Not tested for gradient checking for cross entropy cost function other than sigm layer'
end

ann.CalcLastLayerActDerivative =1;
if ann.loss_func == 'cros' 
    if ann.layers{ann.no_of_layers}.act_func == 'soft'
        ann.CalcLastLayerActDerivative =0;
    elseif ann.layers{ann.no_of_layers}.act_func == 'sigm'
        ann.CalcLastLayerActDerivative =0;
    end    
end

if ann.layers{ann.no_of_layers}.act_func == 'none'
    ann.CalcLastLayerActDerivative =0;
end

display 'training started...'
ann.loss_array=[];
for i=1:no_of_epochs
    tic
    for j=1:batch_size:m
        if m_index==4
            xx = x(:,:,:,j:j+batch_size-1);
        else
            xx = x(:,:,j:j+batch_size-1);
        end
        yy =y(:,j:j+batch_size-1);
        ann=ffann(ann, xx);
        ann = bpann(ann,yy);
        ann =gradientdescentann(ann);
        
        ann.loss_array = [ann.loss_array ann.loss];
    end
    toc
end
plot(1:no_of_epochs*no_of_batches, ann.loss_array)
