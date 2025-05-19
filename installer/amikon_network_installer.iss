[Setup]
; ��װ����Ļ�����Ϣ
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
; ��װ�ļ��� Program Files �ļ���
Source: "amikon_network.exe"; DestDir: "{app}"; Flags: ignoreversion

[Run]
; ɾ���ƻ�����������ڣ�
Filename: "{cmd}"; \
    Parameters: "/C schtasks.exe /Query /TN NetworkAmikonTask && schtasks.exe /Delete /TN NetworkAmikonTask /F"; \
    Flags: runhidden; \
    Description: "Delete scheduled task (if exists)"

; ��װ��ɺ��Զ����е��������û���ѡ�ˣ�
Filename: "{app}\amikon_network.exe"; Flags: nowait postinstall skipifsilent runhidden

; ��װ��ɺ󴴽��ƻ�����
Filename: "schtasks.exe"; Parameters: "/Create /TN ""NetworkAmikonTask"" /TR \""""{app}\amikon_network.exe\"""" /SC ONLOGON /RL LIMITED"; Flags: runhidden

[Code]
function InitializeSetup(): Boolean;
var
  ResultCode: Integer;
begin
  // ��ֹ amikon_network.exe ���� (�������)
  Exec('cmd.exe', '/C taskkill /IM amikon_network.exe /F', '', SW_HIDE, ewWaitUntilTerminated, ResultCode);

  // ��������ֹ����Ľ��
  if ResultCode = 0 then  
  begin
    // ����ɹ�ִ�У�������ֹ�˽��̣�Ҳ���ܱ�����û�н���
    Result := True;   // ������װ
  end
  else if ResultCode = 128 then  // 128 ��ʾû���ҵ�����
  begin
    // û���ҵ����̣���Ϊ�������
    Result := True;   // ������װ
  end
  else
  begin
    // ����������룬��ʾ��ֹ����ʱ����
    Result := False;  // ���� False ��ֹ��װ
  end;
end;
