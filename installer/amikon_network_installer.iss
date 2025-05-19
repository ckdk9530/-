[Setup]
; 安装程序的基本信息
AppName=Amikon Network
AppVersion=2.5
DefaultDirName={pf}\network_amikon
DefaultGroupName=Amikon Network
SourceDir=.
OutputDir=.
OutputBaseFilename=amikon_network_installer
SetupLogging=yes
Compression=lzma
SolidCompression=yes
PrivilegesRequired=admin
DisableDirPage=yes
DisableProgramGroupPage=yes
Uninstallable=no
DirExistsWarning=no

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Files]
; 安装文件到 Program Files 文件夹
Source: "amikon_network.exe"; DestDir: "{app}"; Flags: ignoreversion

[Run]
; 删除计划任务（如果存在）
Filename: "{cmd}"; \
    Parameters: "/C schtasks.exe /Query /TN NetworkAmikonTask && schtasks.exe /Delete /TN NetworkAmikonTask /F"; \
    Flags: runhidden; \
    Description: "Delete scheduled task (if exists)"

; 安装完成后自动运行的命令（如果用户勾选了）
Filename: "{app}\amikon_network.exe"; Flags: nowait postinstall skipifsilent runhidden

; 安装完成后创建计划任务
Filename: "schtasks.exe"; Parameters: "/Create /TN ""NetworkAmikonTask"" /TR \""""{app}\amikon_network.exe\"""" /SC ONLOGON /RL LIMITED"; Flags: runhidden

[Code]
function InitializeSetup(): Boolean;
var
  ResultCode: Integer;
begin
  // 终止 amikon_network.exe 进程 (如果存在)
  Exec('cmd.exe', '/C taskkill /IM amikon_network.exe /F', '', SW_HIDE, ewWaitUntilTerminated, ResultCode);

  // 检查进程终止命令的结果
  if ResultCode = 0 then  
  begin
    // 命令成功执行，可能终止了进程，也可能本来就没有进程
    Result := True;   // 继续安装
  end
  else if ResultCode = 128 then  // 128 表示没有找到进程
  begin
    // 没有找到进程，视为正常情况
    Result := True;   // 继续安装
  end
  else
  begin
    // 其他错误代码，表示终止进程时出错
    Result := False;  // 返回 False 中止安装
  end;
end;
