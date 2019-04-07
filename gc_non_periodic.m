function [spikes] = gc_non_periodic(runs,filename,n,tau,dt,beta,alphabar,abar,wtphase,alpha,useSpiking)
%-----------------------------------
% Grid Cell Dynamics - Periodic
%-----------------------------------



%---------------------
% LOAD AND CLEAN DATA
%---------------------
FileLoad = 0;
if exist(filename,'file') == 2
    
    load(filename)
    FileLoad = 1;
    
else
    % If no data is loaded, use random trajectories.
    % Random head directions between 0 and 2*pi with no more than 20 degree
    % turn at each time step and trajectory based off of the previous time
    % step's head direction
    total = 300000;
    l = 12;
    L = 75;
    enclosureRadius = 2*100; % Two meters
    temp_velocity = rand()/2;
    position_x = zeros(total,1);
    Weights = zeros(total,1);
    position_y = zeros(total,1);
    headDirection = zeros(total,1)';
    
    %{
    position_x_learn = zeros(total,1);
    position_y_learn = zeros(total,1);
    headDirection_learn = zeros(total,1)';
    %}
    position_x(1) = L - 6 + rand()*(6);
    position_y(1) = rand()*L;
    headDirection(1) = rand()*2*pi;
   
    %{
    position_x_learn(1) = rand()*(12)+(L-12);
    position_y_learn(1) = rand()*(12)+(L-12);
    headDirection_learn(1) = rand()*2*pi;
    %}
    
    for i = 2:total
        % max acceleration is .1 cm/ms^2
        temp_rand = max(min(normrnd(0,.05),.2),-.2); 
        
        % max velocity is .5 cm/ms
        temp_velocity = min(max(temp_velocity + temp_rand,0),.25); 
        
        % Don't let trajectory go outside of the boundry, if it would then randomly
        % rotate to the left or right
        leftOrRight = round(rand());
        if (leftOrRight == 0)
            leftOrRight = -1;
        end
        
        %Square environment
        
         while (abs(position_x(i-1) + cos(headDirection(i-1))*temp_velocity)>L || ...
             (abs(position_y(i-1) + sin(headDirection(i-1))*temp_velocity) > L))
           
            headDirection(i-1) = headDirection(i-1) + leftOrRight*pi/100;
           
         end 
        
        %||(abs(position_y(i-1) + sin(headDirection(i-1))*temp_velocity) < L-12||(abs(position_x(i-1) + sin(headDirection(i-1))*temp_velocity) < L-12)))
        %{
        while (sqrt((position_x(i-1) + cos(headDirection(i-1))*temp_velocity)^2 ...
            + (position_y(i-1) + sin(headDirection(i-1))*temp_velocity)^2)  > enclosureRadius)
           
            headDirection(i-1) = headDirection(i-1) + leftOrRight*pi/100;
           
        end
         %}
        position_x(i) = position_x(i-1)+cos(headDirection(i-1))*temp_velocity; 
        position_y(i) = position_y(i-1)+sin(headDirection(i-1))*temp_velocity; 
        headDirection(i) = mod(headDirection(i-1) + (rand()-.5)/5*pi/2,2*pi);
         
         
    end
    
    %rotate coordinates, these are fed into the network
    shift = rand()*2*pi
    headDirection_r = headDirection+shift.*ones(size(headDirection));
    position_x_r = position_x.*cos(shift) + position_y.*sin(shift);
    position_y_r = position_y.*cos(shift) - position_x.*sin(shift);
    
end
b = ['offset',num2str(runs),'.mat'];
save(b, 'shift')
sampling_length = length(position_x);
position = [position_x,position_y];
%{
if FileLoad == 1
%linearly interpolate data to scale to .5 ms
    if dt ~= .5
        if dt < .1
            % If data is too fine, then downsample to make computations faster
            position_x = downsample(position_x,floor(.5/dt));
            position_y = downsample(position_y,floor(.5/dt));
            dt = floor(.5/dt)*dt;
        end
        dt = round(dt*10)/10;
    
        position_x = interp(position_x,dt*10);
        position_y = interp(position_y,dt*10);
    
        position_x = downsample(position_x,5);
        position_y = downsample(position_y,5);
        dt = .5;
    end
    
    sampling_length = length(position_x);
    % Add in head directions
    headDirection = zeros(sampling_length,1);
    for i = 1:(sampling_length - 1)
        headDirection(i) = mod(atan2((position_y(i+1)- position_y(i)),(position_x(i+1) - position_x(i))),2*pi);  
    end
    headDirection(sampling_length) = headDirection(sampling_length-1);
end
%}
%----------------------
% INITIALIZE VARIABLES
%----------------------

% padding for convolutions
big = 2*n; 
dim = n/2; 

% initial population activity
r=zeros(n,n);  
rfield = r; 
s = r;

% A placeholder for spiking activity
spikes = cell(sampling_length,1);
spikes(:) = {sparse(n,n)};

% A placeholder for a single neuron response
sNeuronResponse = zeros(sampling_length,1);
sNeuron = [n/2, n/2];

% Envelope and Weight Matrix parameters
x = -n/2:1:n/2-1; 
lx=length(x);
xbar=sqrt(beta)*x; 

%------------------------------------
% INITIALIZE SYNAPTIC WEIGHT MATRICES
%------------------------------------

% The idea is to view the neural population activity as an input signal - x(t), 
% with the weight matrix - h(t), as an impulse response yeilding the 
% output signal as the new population activity - y(t). To get y(t) we will 
% convolute x(t) with h(t).


% The center surround, locally inhibitory weight matrix - Equation (3)

filt = abar*exp(-alphabar*(ones(lx,1)*xbar.^2+xbar'.^2*ones(1,lx)))...
       -exp(-1*(ones(lx,1)*xbar.^2+xbar'.^2*ones(1,lx)));  

% The envelope function that determines the global feedforward
% input - Equation (5)

venvelope = exp(-4*(x'.^2*ones(1,n)+ones(n,1)*x.^2)/(n/2)^2); 

% We create shifted weight matrices for each preferred firing direction and
% transform them to obtain h(t).

frshift=circshift(filt,[0,wtphase]); 
flshift=circshift(filt,[0,-wtphase]);
fdshift=circshift(filt,[wtphase,0]); 
fushift=circshift(filt,[-wtphase,0]);

ftu=fft2(fushift,big,big); 
ftd=fft2(fdshift,big,big); 
ftl=fft2(flshift,big,big); 
ftr=fft2(frshift,big,big);

% Block matricies used for identifying all neurons of one preferred firing
% direction

typeL=repmat([[1,0];[0,0]],dim,dim);  
typeR=repmat([[0,0];[0,1]],dim,dim);  
typeU=repmat([[0,1];[0,0]],dim,dim);  
typeD=repmat([[0,0];[1,0]],dim,dim); 

%Weights between border cells and grid
W = zeros(n,n,4);
%strength of Hebbian correction
beat = 200;
%learning rate of hebbian 
gamma = 0.002;
 
%----------------------------
% INITIAL MOVEMENT CONDITIONS
%----------------------------

theta_v=pi/5;
left = -sin(theta_v); 
right = sin(theta_v);
up = -cos(theta_v); 
down = cos(theta_v); 
vel=0; 


    %------------------
    % BEGIN SIMULATION 
    %------------------
    fig = figure(1);
    set(fig, 'Position',[50,1000,550,450]);
	

    % We run the simulation for 250 ms with zero velocity

    for iter=1 : 500
        
        %----------------------------------------
        % COMPUTE NEURAL POPULATION ACTIVITY 
        %----------------------------------------
     

        % Break global input into its directional components
        % Equation (4)
        
        
        rfield = venvelope.*((1+vel*right)*typeR+(1+vel*left)*typeL+(1+vel*up)*typeU+(1+vel*down)*typeD);    

        
        % Convolute population activity with shifted semmetric weights.
        % real() is implemented for octave compatibility
        convolution = real(ifft2(...
			   fft2(r.*typeR,big,big).*ftr ...
			 + fft2(r.*typeL,big,big).*ftl ...
			 + fft2(r.*typeD,big,big).*ftd ...
			 + fft2(r.*typeU,big,big).*ftu));         
      
        % Add feedforward inputs to the shifted population activity to
        % yield the new population activity.
        rfield = rfield+convolution(n/2+1:big-n/2,n/2+1:big-n/2);  

        % Neural Transfer Function
        fr=(rfield>0).*rfield;
       
        % Neuron dynamics (Equation 1)
        r_old = r;
        r_new = min(10,(dt/tau)*(5*fr-r_old)+r_old);
        r = r_new;
        
           
    %-----------------------------------------
    % PLOTS
    %-----------------------------------------
        
        if mod(iter,20)==1,
     	  %imagesc(r_new,[0,2]); colormap(hot); colorbar; drawnow;
          %title('Neural Population Activity');
        end  
    end   
    
   
    %-----------------------------------------------------------
    % COMPUTE NEURAL POPULATION ACTIVITY WITH PERIODIC BOUNDARY
    %-----------------------------------------------------------
   
    % increment is the position in the trajectory data. start at 2 to compute velocity
    increment = 2;
    s = r;
    set(fig,'Position',[50,1000,450,900]);

    for iter=1: sampling_length  - 20      
        %feed in rotated coordinates
        theta_v =  headDirection_r(increment);
        vel = sqrt((position_x_r(increment) - position_x_r(increment- 1))^2 + (position_y_r(increment) - position_y_r(increment- 1))^2);
        
        left = -cos(theta_v); 
        right = cos(theta_v);
        up = sin(theta_v);
        down = -sin(theta_v);
                
        increment = increment + 1;
                 

        % Break feedforward input into its directional components
        % Equation (4) 
        %learn hebbian weights, these weights are in actual space
        hebb = 0;
        
        border(1) = position_x(iter)>L-l;
        
        border(2) = position_x(iter)<l-L;
        border(3) = position_y(iter)>L-l;
        border(4) = position_y(iter)<l-L;
        for h =1:4
            hebb = W(:,:,h)*border(h) + hebb;
        end
        
        rfield = venvelope.*((1+alpha*vel*right)*typeR+(1+alpha*vel*left)*typeL+(1+alpha*vel*up)*typeU+(1+alpha*vel*down)*typeD + beat*hebb);    

        
        % Convolute population activity with shifted semmetric weights.
        % real() is implemented for octave compatibility
        convolution = real(ifft2(...
			   fft2(r.*typeR,big,big).*ftr ...
			 + fft2(r.*typeL,big,big).*ftl ...
			 + fft2(r.*typeD,big,big).*ftd ...
			 + fft2(r.*typeU,big,big).*ftu));   
       
        % Add feedforward inputs to the shifted population activity to
        % yield the new population activity.
    

        rfield = rfield+convolution(n/2+1:big-n/2,n/2+1:big-n/2); 
     
        % Neural Transfer Function
        rfield(rfield<0) = 0;
       
       
        % Neuron dynamics (Eq. 1)
        r_old = r;
        r_new = min(10,(dt/tau)*(5*rfield-r_old)+r_old);
        r = r_new;
        
          %hebbian learning weights
        for h = 1:4
            W(:,:,h) = W(:,:,h) + gamma*r*border(h);
            
    
        end
        
        %divisive normalization for hebbian learning stability
        gridcellsum =sum(sum(W));
        
       for h = 1:4
            W(:,:,h) = W(:,:,h)/gridcellsum(:,:,h);
       end
        W(isnan(W)==1)=0;
        %}
        % Track single neuron response
        
        
       if ( rfield(sNeuron(1),sNeuron(2)) > 0 )
            sNeuronResponse(increment) = 1;
        end
          
	if (useSpiking == 1)

		spike = rfield*dt > rand(n,n);
        
        % Neurons decay according to Equation (6)
		s = min(10,s + (dt/tau)*(-s + (tau/dt)*spike));
		r = s;
        spikes{increment,1} = sparse(spike);
		% Track a single neuron response
        sNeuronResponse(increment)  = full(spikes{increment,1}(sNeuron(1),sNeuron(2)));
    end
    Weights(iter) = W(n/2,n/2,1);

        
    %-----------------------------------------
    % PLOTS
    %-----------------------------------------
        
        
        if mod(iter,3000) == 1 && abs(position_x(iter))<L/2&& abs(position_y(iter)<L/2)
             
            
            %subplot(2,1,1)
            %imagesc(r_new,[0,2]); colormap(hot); colorbar; drawnow;
            %title('Neural Population Activity');
            
            a = ['neuralactivity',num2str(runs),'.mat'];
            save(a,'r_new');
            
        end
        
        if mod(iter,20) == 1
            
            %subplot(2,1,2)
            tempx = sNeuronResponse.*position_x;
            tempy = sNeuronResponse.*position_y;
            tempx(tempx == 0) = [];
            tempy(tempy == 0) = [];
            
            %subplot(2,1,2)
           % plot(position_x(1:increment),position_y(1:increment),'-',position_x(increment),position_y(increment),'o', tempx,tempy,'x')
            %title('Single Neuron Response');
            %axis([min(position_x),max(position_x),min(position_y),max(position_y)]);
            %drawnow;
            
           %{
           fig2 = figure(2);
           set(fig2,'Position',[1050,1000,450,900]);
           subplot(1,1,1);
           imagesc(W(:,:,1));
            %}
        end
            
    end        
    %{
    save('SNR8border50.mat','sNeuronResponse');
    save('savetrajectory8border50.mat','position');
   save('Weightsborder50.mat','Weights');
%}
    
    
end
