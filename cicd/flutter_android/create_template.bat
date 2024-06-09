@echo off
setlocal EnableDelayedExpansion

:: ユーザーからの入力を受け取る
echo Enter the Project Name in lower-kebab-case (e.g., my-project):
set /p ProjectNameKebab=Project Name: 
echo Enter the Default Branch Name (e.g., develop):
set /p DefaultBranchName=Default Branch Name: 
echo Enter the Branch Name that triggers CI/CD (e.g., release):
set /p TriggerBranchName=Trigger Branch Name: 
echo Enter the Flutter Version (e.g., 3.13.4):
set /p FlutterVersion=Flutter Version: 

:: YAMLファイルのパスを指定
set TemplateFile=template.yaml
set YAMLFile=%ProjectNameKebab%-cicd.yaml

:: テンプレートファイルを読み込んでプレースホルダーを置き換える
(
    for /f "tokens=*" %%i in (%TemplateFile%) do (
        set line=%%i
        set line=!line:{{PROJECT_NAME_KEBAB}}=%ProjectNameKebab%!
        set line=!line:{{DEFAULT_BRANCH_NAME}}=%DefaultBranchName%!
        set line=!line:{{TRIGGER_BRANCH_NAME}}=%TriggerBranchName%!
        set line=!line:{{FLUTTER_VERSION}}=%FlutterVersion%!
        echo !line!
    )
) > %YAMLFile%

echo YAML file created as %YAMLFile%

endlocal