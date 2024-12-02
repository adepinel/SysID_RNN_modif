clear all
close all
clc

% Add subfolders to path
folder = fileparts(which(mfilename)); 
addpath(genpath(folder));

%simulatore

%carico il modello
% load ARXpower4              %modello della potenza
% load OETcond4d               
% load ARXToutrefr4
load ARerrss
% load ARXtwrss
% load reg_torri              %PI torri              
% load tf_Tinc_Toutc
%overall model
load ARXpowerov
load ARXpowerov_noinv
load OEToutcondov
load OEToutcondov_noinv
%carico le variabili ausiliarie
load dati_utenze.mat                
load dati_portate.mat                       %PORTATE con tolti valori sotto i 100m^3/h e ridefiniti


%carico i dati
load('dataset_GF1_clean.mat');
load('dataset_GF2_clean.mat');
load('dataset_GF3_clean.mat');
load('dataset_GF4_clean.mat');
load('dataset_GFgenerali_clean.mat');
load('dataset_torrievaporative_clean.mat'); 
load('dataset_assorbitore_clean.mat');
load date_time
load DatiAssorbitore.mat
load cog_master_calc_pot_per_ass_ET_DISP_ASS.mat
load dataset_PHOT_assorbitore_clean.mat

%PARAMETRI:

Ts=60;                      % IL SAMPLING TIME DEVE ESSERE ESPRESSO IN SECONDI
% Tstart  =8.1e5;             % initial time for the simulation
% Tend    =8.2e5;             % final time for the simulation

%overall period in which the data make sense
%Tstart  =7.3e5;             % initial time for the simulation
%Tend    =9.7e5;             % final time for the simulation

%July roughly
% Tstart= 8.06e5;
% Tend=   8.16e5;

Tstart=7.3e5;
Tend=9.7e5;

% Tstart=7.3e5+4.6e4;
% Tend=7.3e5+5.6e4;



cwater=4.186/3600;          %[kJ/(kg*K) * h/s]

wch=355;                    %[m^3/h] - nominal 355 m^3/h
wass=310;                   %[m^3/h] - nominal 310 m^3/h
rho=1000;                   %[kg/m^3]                  
% wch=285;
% wass=255;

%limits of the bypass and the overflow of Chillers!
lowb_Wbp=0;
upb_Wbp=1000;

%bounds for the saturation blocks
lowb_Woff=0;
upb_Woff=1800;
lowerb_whot_est=200;

%Temperature delta's
DToutass=0.1;
Dsetpoint=0.3;

Setpoint_acqua_torri=24;                            %valore per il regolatore torri
threshold_ass_ONOFF=-0.2;

%parameters for the simulations
L=length(COG_GF4_CDW_FLOW(Tstart:Tend));
time=Ts:Ts:L*Ts;
time=time';

statoass = statusAss2;
%the idea is to cut-off the noise
% statoass= zeros  (max(size(COG_ASS1_T_US_REF_T(Tstart:Tend))),1);
% accesoass= find( COG_ASS1_T_US_REF_T(Tstart:Tend) - COG_ASS1_T_IN_REF_T(Tstart:Tend) < threshold_ass_ONOFF);          %accensione definita quando la temperatura di uscita è minore di quella di ingresso
% statoass(accesoass)=1;
% statoass(173376:215263)=0;
% statusAss2=statoass;
% status absorber
% statusAss2=ones(240001,1);
%statusAss2(173376:215263)=0;
%statusAsstON = timeseries(statusAss2,time);

%%%%%%%%%%%%%%%%%%%%
%%% bypass filter
%%%%%%%%%%%%%%%%%%%%

%tau_wbp=240;                    %[s]
%tau_Tman=60;                    %[s]
%filterbp=tf(0.95, conv([tau_wbp 1], [tau_wbp 1]));

% filterd_bp=c2d(filterbp,Ts,'tustin')
% num_filter_bp=cell2mat(filterd_bp.num);
% den_filter_bp=cell2mat(filterd_bp.den);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%filtering of the temperature & finding initial condition
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tau_Tman=60;                %[s]
%filterTman=tf(1,conv([tau_Tman 1],[tau_Tman 1]));
filterTman=tf(1,[tau_Tman 1]);


%filterd_Tman=c2d(filterTman,Ts,'tustin')           %this one allows for a direct feedthrough
filterd_Tman=c2d(filterTman,Ts);                    %this is strictly proper
filterdTmanss=ss(filterd_Tman);

time_filt_Tman=0:Ts:(length(COG_GF_TM_CF)-1)*Ts;
Tman_filt=lsim(filterd_Tman,COG_GF_TM_CF,time_filt_Tman);

num_filter_Tman=cell2mat(filterd_Tman.num);
den_filter_Tman=cell2mat(filterd_Tman.den);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% build vectors to be passed to the Simulink blocks
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
buildtot=buildtot(Tstart:Tend);
weektime=weektime(Tstart:Tend);
wendtime=wendtime(Tstart:Tend);

% load flow data
whot=whot(Tstart:Tend);                 %treated as an external disturbance

statotwr4=ones(max(size(COG_TE_REG_INV_TE4A(Tstart:Tend))),1);
spentotwr4=find(COG_TE_REG_INV_TE4A(Tstart:Tend)<=0);
statotwr4(spentotwr4)=0;

% individuazione istanti stato acceso (in z) stato acceso = stato= 1, stato spento = stato = 0
stato4=ones(max(size(COG_GF4_LINE_KW(Tstart:Tend))),1);
spento4=find(COG_GF4_LINE_KW(Tstart:Tend)<=0);
stato4(spento4)=0;

stato1=ones(max(size(COG_GF1_LINE_KW(Tstart:Tend))),1);
spento1=find(COG_GF1_LINE_KW(Tstart:Tend)<=0);
stato1(spento1)=0;

stato2=ones(max(size(COG_GF2_LINE_KW(Tstart:Tend))),1);
spento2=find(COG_GF2_LINE_KW(Tstart:Tend)<=0);
stato2(spento2)=0;

stato3=ones(max(size(COG_GF3_LINE_KW(Tstart:Tend))),1);
spento3=find(COG_GF3_LINE_KW(Tstart:Tend)<=0);
stato3(spento3)=0;



%definition of the setpoint vector
setpoint=8;

setp=ones(max(size(COG_GF4_LINE_KW(Tstart:Tend))),1) *setpoint;
errore4_Tin=setp-COG_GF4_ECW(Tstart:Tend);

% SELECT subset of data
Toutrefr4   =COG_GF4_LCW(Tstart:Tend);
Toutcond4   =COG_GF4_LCDW(Tstart:Tend);
Tinrefr4    =COG_GF4_ECW(Tstart:Tend);
Tincond4    =COG_GF4_ECDW(Tstart:Tend);
Power4      =COG_GF4_LINE_KW(Tstart:Tend);
Text        =COG_TE_T_EST(Tstart:Tend);
Inverter4   =COG_TE_REG_INV_TE4A(Tstart:Tend);
Hext        =COG_TE_UMID_EST(Tstart:Tend);

Toutrefr1   =COG_GF1_LCW(Tstart:Tend);
Toutcond1   =COG_GF1_LCDW(Tstart:Tend);
Tinrefr1    =COG_GF1_ECW(Tstart:Tend);
Tincond1    =COG_GF1_ECDW(Tstart:Tend);
Power1      =COG_GF1_LINE_KW(Tstart:Tend);
Inverter1   =COG_TE_REG_INV_TE1A(Tstart:Tend);

Toutrefr2   =COG_GF2_LCW(Tstart:Tend);
Toutcond2   =COG_GF2_LCDW(Tstart:Tend);
Tinrefr2    =COG_GF2_ECW(Tstart:Tend);
Tincond2    =COG_GF2_ECDW(Tstart:Tend);
Power2      =COG_GF2_LINE_KW(Tstart:Tend);
Inverter2   =COG_TE_REG_INV_TE2A(Tstart:Tend);

Toutrefr3   =COG_GF3_LCW(Tstart:Tend);
Toutcond3   =COG_GF3_LCDW(Tstart:Tend);
Tinrefr3    =COG_GF3_ECW(Tstart:Tend);
Tincond3    =COG_GF3_ECDW(Tstart:Tend);
Power3      =COG_GF3_LINE_KW(Tstart:Tend);
Inverter3   =COG_TE_REG_INV_TE3A(Tstart:Tend);

Tman_GF     =COG_GF_TM_CF(Tstart:Tend);          %Temperatura mandata generale 
Trit_GF     =COG_GF_TR_CF(Tstart:Tend);          %Temperatura ritorno generale
Thot_utilities=Thot_utilities(Tstart:Tend);      %temperatura utenze con le temperature delle singole utenze
Thot_pipes=Thot_pipes(Tstart:Tend);      %temperatura utenze: media pesata dei collettori parziali di ritorno, simili a quello a valle
Tcold_pipes=Tcold_pipes(Tstart:Tend);      %temperatura utenze: media pesata dei collettori parziali di mandata, simili a quello a valle
Ton_in=Ton_in(Tstart:Tend);
Ton_out=Ton_out(Tstart:Tend);

date_time=DATETIME(Tstart:Tend);

%variabili dell'assorbitore
Tinass= COG_ASS1_T_IN_REF_T(Tstart:Tend);
Toutass= COG_ASS1_T_US_REF_T(Tstart:Tend) + DToutass;
Toutass_Q= COG_ASS1_T_US_REF_T(Tstart:Tend);

Qass=(Toutass-Trit_GF).*(rho*cwater*wass).*statoass;
Qass2=(Toutass-Tinass).*(rho*cwater*wass).*statoass;

%figure;plot(Tinass);hold on; plot(Trit_GF);legend('Tinass','Tcollrit')


setpoint_ass=7.5;                                %preso dai dati storici (COG_ASS1_SP_T_REFF_T) 

%figure;plot(Tinass); hold on; plot(Toutass); plot(Trit_GF); legend('Tinass','Toutass','Tritorno');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%simulazione modello utenze per profilo iniziale
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%load('Qt_ut');

% definizione condizioni iniziali
Power4_0 = COG_GF4_LINE_KW(Tstart);
Toutrefr4_0 = COG_GF4_LCW(Tstart);
Toutcond4_0 = COG_GF4_LCDW(Tstart);
Touttwr_0 = COG_GF4_LCDW(Tstart);

%figure;plot(statoass);grid on; title('Stato accensione assorbitore');

statotot=stato1+stato2+stato3+stato4;
statotot_ass=statotot+statoass;
ass_off_instants=length(find(statotot_ass==0));

%OPTIMIZATION VARIABLES must be defined as global ones
global Dwell_time
Dwell_time=10;
global DT_ON
DT_ON=0;
global DT_OFF
DT_OFF=0;




%parametri di regolazione
KP=-20;
KI=-1.5e-5;

%parametri PI assorbitore
KPass1 = 0.1;
KIass1 = 5.0000e-04;
KPass2 = 66;
KIass2 = 0.35;

% dinamica assorbitore
s = tf('s');
% Dyn_abs = (1/400)/(s+1/400);
Dyn_abs = (1/180)/(s+1/180);
[Z,poli,k] = zpkdata(Dyn_abs);


Dyn_abs_disc = c2d(Dyn_abs,Ts,'tustin');
num_dyn_abs = Dyn_abs_disc.Numerator{1,1};
den_dyn_abs = Dyn_abs_disc.Denominator{1,1};


% scambio Text-pipes
k_pipes=12.5;                  %k * (Text- T) = Q_pipes



%%%%% new

Qhot = zeros(length(cog_master_calc_pot_per_ass_ET_DISP_ASS(Tstart:Tend)),1);
for k = 0:length(cog_master_calc_pot_per_ass_ET_DISP_ASS(Tstart:Tend))-1
    if cog_master_calc_pot_per_ass_ET_DISP_ASS(Tstart+k)>0
        Qhot(k+1)=cog_master_calc_pot_per_ass_ET_DISP_ASS(Tstart+k);
    end 
end

Qhotd = timeseries(Qhot,time);
Qhot = timeseries(cog_master_potenza_caldo_GF_5_ET_ASS(Tstart:Tend),time);
figure;plot(statusAsst2);
figure;plot(Qhotd);

load buildtot2.mat
buildtot2=buildtot2(Tstart:Tend);

global soglia
soglia = 3.6279;

global gain
gain = 1.4940;

global b0;
b0=0.6303;
global b1;
b1 = -4.3826;
global b2;
b2 = 14.1582;
global p;
p= 0.1909;

% global b0;
% b0=0.2809;
% global b1;
% b1 = -0.0959;
% global b2;
% b2 = 1.0426;
% global b01;
% b01=0.4112;
% global b11;
% b11 = 0.7355;
% global b21;
% b21 = -0.9510;
% global p;
% p= 0.3107;

QHOT = cog_master_calc_pot_per_ass_ET_DISP_ASS(Tstart:Tend);
eta = -QHOT./Qass;
for i = 1:length(eta)
    if eta(i) > 1 
        eta(i)=1;
    end
    if eta(i) < -1 
        eta(i)=-1;
    end
end
figure;plot(eta)

KPass = 66;
KIass = 0.35;

%#####################################################################
%% extracting values for Python
%####################################################################
load absroberstatus.mat
load Qu.mat
one_week = 231840 - 221760;
nb_weeks = 3; % for the switch case
val_middle = 1==1; % do we want validation set tp be in the middle ? 
switch nb_weeks
    case 1 
        start = 4*one_week; 
        end_train = 12*one_week; %debut4 --> fin11
        end_val = 13*one_week;  % debut 12--> fin 12
    case 2
        start = 4*one_week;      
        end_train = 10*one_week; %debut4 --> fin9
        end_val = 11*one_week;  % debut 9--> fin 9
        one_week = 2*one_week;  %change the number of weeks per nExp
        
    case 3
        start = 4*one_week;      
        end_train = 10*one_week; %debut4 --> fin9
        end_val = 11*one_week;  % debut 9--> fin 9
        one_week = 3*one_week;  %change the number of weeks per nExp
    case 4
        start = 4*one_week;      
        end_train = 12*one_week; %debut4 --> fin11
        end_val = 13*one_week;  % debut 12--> fin 12
        one_week = 4*one_week;  %change the number of weeks per nExp 
    case 6
        start = 4*one_week;      
        end_train = 10*one_week; %debut4 --> fin9
        end_val = 11*one_week;  % debut 9--> fin 9
        one_week = 6*one_week;  %change the number of weeks per nExp       
end 
nb_training_sets = (end_train - start)/one_week;

%% clean data
find(Trit_GF == 8);

Trit_GF(74471)= 11;
%% ######################General data#############################
Qhot_valve_dpd = Qhotd.Data.*statusAsst.Data; %absorberstatus.Data ; %Qabsorber with status 
% separated 
%Qh = Qhotd.Data;
%SV = absorberstatus.Data; %statusAsst2.Data;

WEEK = weektime + wendtime./2;%time of the week
Status_global = stato1 + stato2 + stato3 + stato4; % chillers status
time_sep = [weektime, wendtime]; %for MLP
time__ = time(1:one_week);
Toutchillers = (Toutrefr1 + Toutrefr2 + Toutrefr3 + Toutrefr4)/4; %mean of the Toutrefr

%% ######################Training data###############################
%nb_weeks_train = 2;
%get the training data
for i = 1:nb_training_sets
    Qhot_train = Qhot_valve_dpd(start + (i-1)*one_week + 1:start + one_week*i); %absorber
    Week_train = WEEK(start + (i-1)*one_week + 1:start + one_week*i); %time of the week
    Statusglobal_train = Status_global(start + (i-1)*one_week + 1:start + one_week*i); %chillers status
    % T_ext and H_ext inputs
    Text_train = Text(start + (i-1)*one_week + 1:start + one_week*i); 
    Hext_train = Hext(start + (i-1)*one_week + 1:start + one_week*i); 
    % OUTPUTS : Trit and Tmanifold
    Trit_GF_train = Trit_GF(start + (i-1)*one_week + 1:start + one_week*i);  
    Tman_GF_train = Tman_GF(start + (i-1)*one_week + 1:start + one_week*i); 
    
    %normalize data
    dExp{i} = [Text_train./max(Text) Hext_train./max(Hext) Week_train./max(WEEK) Statusglobal_train./3 Qhot_train./max(Qhot_valve_dpd)];
    yExp{i} = [Trit_GF_train./max(Trit_GF) Tman_GF_train./max(Tman_GF)];

    % for testing with Qhot and statusAsst separatly
    %Qh_train = Qh(start + (i-1)*one_week + 1:start + one_week*i);
    %sv_train = SV(start + (i-1)*one_week + 1:start + one_week*i);
    %separate{i} = [sv_train./max(sv_train) Qh_train./max(Qh_train)]; 
end
%buildtotnorm = Qu.Data(start+1:end_train)./max(Qu.Data);
buildtotnorm = buildtot(start+1:end_train)./max(buildtot);
time_separated = time_sep(start+1:end_train,:)./max(time_sep(start+1:end_train,:));
Toutass_train = Toutass(start+1:end_train)./max(Toutass);
Toutchillers_train = Toutchillers(start+1:end_train)./max(Toutchillers);
%% ##############Validation data #####################################
Qhot_val = Qhot_valve_dpd(end_train +1:end_val);
%Qh_val = Qh(end_train +1:end_val);
%sv_val = SV(end_train +1:end_val);

time_separated_val = time_sep(end_train+1:end_val,:)./max(time_sep(end_train+1:end_val,:));
Week_val = WEEK(end_train +1:end_val);

Statusglobal_val = Status_global(end_train +1:end_val);
time_val = time(end_train +1:end_val);

Text_val = Text(end_train+1:end_val);
Hext_val = Hext(end_train+1:end_val);

Trit_GF_val = Trit_GF(end_train+1:end_val);
Tman_GF_val = Tman_GF(end_train+1:end_val);

%buildtotnorm_val = Qu.Data(end_train+1:end_val)./max(Qu.Data);
buildtotnorm_val = buildtot(end_train+1:end_val)./max(buildtot);
Toutass_val = Toutass(end_train+1:end_val)./max(Toutass);
Toutchillers_val = Toutchillers(end_train+1:end_val)./max(Toutchillers);
%separate_val = {[sv_val./max(sv_val) Qh_val./max(Qh_val)]};

%make one variable for in and output
dExp_val = {[Text_val./max(Text) Hext_val./max(Hext) Week_val./max(WEEK) Statusglobal_val./3 Qhot_val./max(Qhot_valve_dpd)]}; 
yExp_val = {[Trit_GF_val./max(Trit_GF) Tman_GF_val./max(Tman_GF)]};


%% get the training data again IF val_middle is true !
if (nb_weeks == 1)
    nb_added_weeks = 4;
elseif (nb_weeks == 3)
    nb_added_weeks = 2;
elseif (nb_weeks == 2) 
    nb_added_weeks = 3;
else 
    nb_added_weeks = 1;
end


if val_middle== 1 
    start = end_val;
    for i = 1:nb_added_weeks % number of weeks for training after validation set
        Qhot_train = Qhot_valve_dpd(start + (i-1)*one_week + 1:start + one_week*i);
        %Qh_train = Qh(start + (i-1)*one_week + 1:start + one_week*i);
        %sv_train = SV(start + (i-1)*one_week + 1:start + one_week*i);
        
        Week_train = WEEK(start + (i-1)*one_week + 1:start + one_week*i);
        
        Statusglobal_train = Status_global(start + (i-1)*one_week + 1:start + one_week*i);
        
        Text_train = Text(start + (i-1)*one_week + 1:start + one_week*i); 
        Hext_train = Hext(start + (i-1)*one_week + 1:start + one_week*i); 
        
        Trit_GF_train = Trit_GF(start + (i-1)*one_week + 1:start + one_week*i);  
        Tman_GF_train = Tman_GF(start + (i-1)*one_week + 1:start + one_week*i); 
        
        %normalize data
        dExp{i+nb_training_sets} = [Text_train./max(Text) Hext_train./max(Hext) Week_train./max(WEEK) Statusglobal_train./3 Qhot_train./max(Qhot_valve_dpd)];
        yExp{i+nb_training_sets} = [Trit_GF_train./max(Trit_GF) Tman_GF_train./max(Tman_GF)];
        % for testing with Qhot and statusAsst or valve separatly
        %separate{i+nb_weeks_train+1} = [sv_train./max(sv_train) Qh_train./max(Qh_train)]; 
    end
    %buildtotnorm = [buildtotnorm ; Qu.Data(start+1:start + one_week*nb_added_weeks)./max(Qu.Data)];
    buildtotnorm = [buildtotnorm ; buildtot(start+1:start + one_week*nb_added_weeks)./max(buildtot)];
    Toutass_train = [Toutass_train ; Toutass(start+1:start + one_week*nb_added_weeks)./max(Toutass)];
    Toutchillers_train = [Toutchillers_train ; Toutchillers(start+1:start + one_week*nb_added_weeks)./max(Toutchillers)];
    time_separated = [time_separated ; time_sep(start+1:start + one_week*nb_added_weeks,:)./max(time_sep(start+1:start + one_week*nb_added_weeks,:))];
end 
%% 
size(dExp)

maxTrit = max(Trit_GF);
maxTman = max(Tman_GF);

plot(date_time,Trit_GF);
 title('valTemperature returning from the building')
 ylabel('Temperature[°C]')
 figure;

 plot(Toutass_val);
 title('valTemperature returning from the absorber')
 ylabel('Temperature[°C]')
 figure;
 plot(Toutchillers_val);
 title('val Temperature returning from the chillers')
 ylabel('Temperature[°C]')
 figure;

 % 12000 - 17000 seems nice for Toutass_train and Toutchillers_train
%% export input and output 
save input_1.mat dExp dExp_val time__
save output_1.mat yExp yExp_val
save subsystems.mat Toutass_train Toutchillers_train Toutass_val Toutchillers_val
save output_Q_1.mat yExp yExp_val buildtotnorm buildtotnorm_val time_separated time_separated_val 
save denormalize maxTrit maxTman
%save input_separate.mat separate separate_val 
%% plotting to see what is  inputs
% figure;
% plot(date_time,Qhotd.Data);
% title('Thermal load of the absorber across time')
% ylabel('Thermal load from absorber [W]')
% figure;
% plot(date_time,statusAsst2.Data);
% ylabel('On/off status of the absorber valve')
% title('Status of the valve opening for the absorber')
% plot(date_time,Status_global);
% ylabel('On/off status of the 4 chillers')
% title('Status of the valve opening for the 4 chillers')
% figure;
% plot(date_time(1:25000),weektime(1:25000));
% title('Time advancing through the days of the week')
% ylabel('Number of minutes since the beginning of the day')
% figure;
% plot(date_time(1500:25000),wendtime(1500:25000));
% title('Time advancing through the weekend')
% ylabel('Number of minutes since the beginning of the weekend')
% 
% plot(date_time,Text);
% title('Temperature measured outside the building')
% ylabel(' Outside Temperature[°C]')
% figure;
% plot(date_time,Hext);
% ylabel('Outside Humidity')
% title('Humidity level measured outside the building')
% 
% %% plotting to see what is outputs
% figure;
% plot(date_time,Trit_GF);
% title('Temperature returning from users')
% ylabel('Temperature [°C]')
% figure;
% 
% plot(date_time,Tman_GF);
% ylabel('Temperature [°C]')
% title('Temperature arriving from manifolds to users')
% 
% 
% figure;
% hold on
% plot(statusAsst2)
% %plot(absorberstatus-6.5)
% %%not the same... plot(statusAsst)
% title('Status of the absorber valve (on/off)')
% hold off
% %xlabel('Time steps')

save Qu.mat Qu
save absroberstatus.mat absorberstatus
%% data to get out (previous)
%input_NN_week_ = transpose(input_NN_week.Data);
%input_NN_we_ = input_NN_we.Data;
%Qout_tot_ = Qout_tot.Data;
%save in_week.mat input_NN_week_ buildtot Qout_tot_
%save in_weekend.mat input_NN_we_
% 
% 
% statusvalve = StatusValve.Data;
% 
% writematrix(statusvalve,'ValveStatus.csv')
% writematrix(Tcold_pipes,'Tcold_pipes.csv')
% writematrix(Thot_pipes,'Thot_pipes.csv')
% writematrix(Tinass,'Tinass.csv')
% writematrix(Toutass,'Toutass.csv')
% 
% Tinrefr = [Tinrefr1 , Tinrefr2, Tinrefr3, Tinrefr4];
% Toutrefr = [Toutrefr1 , Toutrefr2, Toutrefr3, Toutrefr4];
% 
% 
% writematrix(Tinrefr,'Tinrefr.csv')
% writematrix(Toutrefr,'Toutrefr.csv')
% 
% Qhotd_no_timeseries = Qhotd.Data;
% writematrix(Qhotd_no_timeseries, 'Qhotd.csv')

% Define matrix W2 as a row vector (transposed)
W2 = [-0.0212243887164881954443540479360308381728827953338623046875, ...
    -0.1545373718515246352911418625808437354862689971923828125, ...
    -1.178518840988839855299374903552234172821044921875, ...
    -0.46448834744351419345775866531766951084136962890625, ...
    1.094018388292422372387591167353093624114990234375, ...
    -2.310335652593654476305573552963323891162872314453125, ...
    0.0716274140098256129061127239765482954680919647216796875, ...
    3.141668695647897191491892954218201339244842529296875, ...
    0.308801338319257212550184021893073804676532745361328125, ...
    6.3597376477466998068166503799147903919219970703125, ...
    0.12172366468979399500849325477247475646436214447021484375, ...
    0.205653932324065757963893474880023859441280364990234375, ...
    0.040299873480876526044003327342579723335802555084228515625, ...
    -2.973827520961939807619955900008790194988250732421875, ...
    -0.01385167521104509365248436125739317503757774829864501953125, ...
    3.157601673983476242568713132641278207302093505859375, ...
    0.08513774173160715275088250564294867217540740966796875, ...
    -1.066587556015377469265104082296602427959442138671875, ...
    -0.262724240328694680357557444949634373188018798828125, ...
    -0.1274241508913905585043124801813974045217037200927734375];

% Define matrix W1 as a 20x3 matrix
W1 = [
    3.80031263036922251075111489626578986644744873046875, 3.48124795309309842394895895267836749553680419921875, 5.6832134476564650782393073313869535923004150390625;
    3.03236744436673166802620471571572124958038330078125, -3.862363025416326589578375205746851861476898193359375, 5.36753221692721016466975925141014158725738525390625;
    -2.2301572506815023899662264739163219928741455078125, 1.42693620719646929728696704842150211334228515625, 4.478758820873057544531548046506941318511962890625;
    1.6609516721205686007323265585000626742839813232421875, 0.397499547847143974710348857115604914724826812744140625, 2.703549969404032804476400997373275458812713623046875;
    -2.237154510209629076911141964956186711788177490234375, 1.545135669545870005237020450294949114322662353515625, 4.35850231597891646373454932472668588161468505859375;
    -0.71344631677347114528942029210156761109828948974609375, -0.346768482644448761131883429698063991963863372802734375, -0.63766168610719209386417105633881874382495880126953125;
    -4.4109267123779343222622628672979772090911865234375, -1.2398871048944000161640133228502236306667327880859375, 0.08874734064305182801302152029165881685912609100341796875;
    0.261376879669295292263342389560420997440814971923828125, 0.046983457719907535665715414552323636598885059356689453125, -14.65424888280293913567220442928373813629150390625;
    1.0834736151117467795046422907034866511821746826171875, -0.54712206076450276004408124208566732704639434814453125, -2.91566518493582815807485530967824161052703857421875;
    0.196326979487118691292124594838242046535015106201171875, 0.040203229066089814225382070844716508872807025909423828125, 14.74818455625534596720171975903213024139404296875;
    -1.37993429452527749390355893410742282867431640625, 1.80787092956229766826936611323617398738861083984375, -1.34962624038843603813120353152044117450714111328125;
    0.2366101784710636923847459911485202610492706298828125, -0.79760279068770756349948669594596140086650848388671875, 3.31878018652791784148803344578482210636138916015625;
    -6.17576069834932273039385108859278261661529541015625, -4.6213766153946238546268432401120662689208984375, -0.53212311119722677243970565541530959308147430419921875;
    0.63142906967199274870239378287806175649166107177734375, 0.1383314423053734476543041864715632982552051544189453125, 15.8517462256957539779023136361502110958099365234375;
    5.10549275828433479773593717254698276519775390625, 12.2655634984556431987812175066210329532623291015625, -1.0745946811867785886107640180853195488452911376953125;
    -0.0206029857737932732553165493527558282949030399322509765625, -0.52408786463345069517316687779384665191173553466796875, -1.224788998822464503035689631360583007335662841796875;
    3.9240529804532560120833295513875782489776611328125, 1.9384978617504209363886502615059725940227508544921875, 7.31616909649334434817546934937126934528350830078125;
    -0.2834253700128750441677993876510299742221832275390625, -1.1381428168690630453596668303362093865871429443359375, -1.03774142249906464741115996730513870716094970703125;
    0.99290145934405116268095525811077095568180084228515625, 0.09200010600391918258456058765659690834581851959228515625, -13.19145646578102315515934606082737445831298828125;
    4.7592818387215505282483718474395573139190673828125, 0.10889219085553902832685935209156014025211334228515625, -1.1007803599125327576047084221499972045421600341796875
];

b1 = [-6.67262814217586264220472003216855227947235107421875;-7.3108706230655204905133359716273844242095947265625;3.4039251581715088690316406427882611751556396484375;-2.516850789448188407959605683572590351104736328125;3.272386618660166224259455702849663794040679931640625;0.79864617823052841405484514325507916510105133056640625;0.8366002177527318561800484530976973474025726318359375;-8.59127454397133050179036217741668224334716796875;-0.410086575549143128593954088501050136983394622802734375;8.93911633607626043840355123393237590789794921875;0.51200626880868049806139197244192473590373992919921875;-0.52474830173068831573601755735580809414386749267578125;0.66644940536927810281753181698150001466274261474609375;9.8818115736700153917126954183913767337799072265625;5.66685987139126012834822176955640316009521484375;1.4876837454747644517993876434047706425189971923828125;2.2077538134991385732064372859895229339599609375;1.5363128151006566479708226324873976409435272216796875;12.0813189733423644867116308887489140033721923828125;3.423189196682755675027465258608572185039520263671875];
b2 = -0.80830390220496706188413327254238538444042205810546875;
save W.mat W1 W2 b1 b2
