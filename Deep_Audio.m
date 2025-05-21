function varargout = Deep_Audio(varargin)
% DEEP_AUDIO MATLAB code for Deep_Audio.fig
%      DEEP_AUDIO, by itself, creates a new DEEP_AUDIO or raises the existing
%      singleton*.
%
%      H = DEEP_AUDIO returns the handle to a new DEEP_AUDIO or the handle to
%      the existing singleton*.
%
%      DEEP_AUDIO('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in DEEP_AUDIO.M with the given input arguments.
%
%      DEEP_AUDIO('Property','Value',...) creates a new DEEP_AUDIO or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before Deep_Audio_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to Deep_Audio_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help Deep_Audio

 

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @Deep_Audio_OpeningFcn, ...
                   'gui_OutputFcn',  @Deep_Audio_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before Deep_Audio is made visible.
function Deep_Audio_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to Deep_Audio (see VARARGIN)

% Choose default command line output for Deep_Audio
handles.output = hObject;
axes(handles.axes1); axis off
axes(handles.axes2); axis off
    data=get(handles.uitable1,'data');
    data{1,1}='';
    data{1,2}='';
    data{1,3}='';
    data{1,4}='';
    data{1,5}='';
    set(handles.uitable1,'data',data);
% Update handles structure
guidata(hObject, handles);

% UIWAIT makes Deep_Audio wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = Deep_Audio_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on button press in pushbutton5.
function pushbutton5_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton5 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
run('Voice_Record.m');

% --- Executes on button press in pushbutton6.
function pushbutton6_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton6 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
axes(handles.axes1); cla(handles.axes1); title(''); axis off
axes(handles.axes2); cla(handles.axes2); title(''); axis off

    data=get(handles.uitable1,'data');
    data{1,1}='**';
    data{1,2}='**';
    data{1,3}='**';
    data{1,4}='**';
    data{1,5}='**';
    set(handles.uitable1,'data',data);

% --- Executes on button press in pushbutton7.
function pushbutton7_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton7 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
close  all

% --- Executes on button press in pushbutton1.
function pushbutton1_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global y1;global fs1;
[fname path]=uigetfile('*.*','Browse Audio');
if fname~=0
[y1,fs1]=audioread([path,fname]);
% [x, fs] = audioread(Audio);
% audioData=getaudiodata(Audio);
plot(handles.axes1,y1);
    sound(y1);
else
    warndlg('Please Select the necessary Image File');
end


% --- Executes on button press in pushbutton2.
function pushbutton2_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
 global y1;global fs1;global range1;
    addpath('voicebox');
    t1=melcepst(y1,fs1);
    mel1=t1(:,1);
    mean1=mean(mel1);
    std1=std(mel1);
    min1=min(mel1);
    max1=max(mel1);
    range1=max1-min1;
    data=get(handles.uitable1,'data');
    data{1,1}=mean1;
    data{1,2}=std1;
    data{1,3}=min1;
    data{1,4}=max1;
    data{1,5}=range1;
    set(handles.uitable1,'data',data);
    plot(handles.axes2,mel1);
    

% --- Executes on button press in pushbutton3.
function pushbutton3_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% global y1;global fs1;global x;global fs;global xEst;
 global y1;global fs1;global x;global fs;global xEst;
x=y1; fs=fs1;
frmSz = 256;
shiftAmount = 64;
fftRes = 1;
xlen = length(x);
x = padarray(x, ceil(xlen/shiftAmount)*shiftAmount - xlen, 0, 'post');
X = HNN(x, frmSz, shiftAmount, fftRes);
Mag = abs(X);
Phase = angle(X);
[Mag2] = HNN_ANALYSIS(Mag, 20);
EstX = Mag2.*exp(Phase*1i);
xEst = real(IHNN(EstX, frmSz, shiftAmount));
[icondata,iconcmap] = imread('1.png');
h=msgbox('MFCC Extraction Done',...
         'Analysis','custom',icondata,iconcmap);


% --- Executes on button press in pushbutton4.
function pushbutton4_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
 global x;global fs;global xEst;
% audiowrite('audio\Nnoisy.wav', x, fs);
% audiowrite('audio\Estimated.wav', xEst, fs);
sound(xEst, fs);
run('Analysis.p');


% --- Executes on selection change in listbox1.
function listbox1_Callback(hObject, eventdata, handles)
% hObject    handle to listbox1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns listbox1 contents as cell array
%        contents{get(hObject,'Value')} returns selected item from listbox1


% --- Executes during object creation, after setting all properties.
function listbox1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to listbox1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: listbox controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
