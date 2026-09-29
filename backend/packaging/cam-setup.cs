// VTM Spark Camera Setup.exe - adds the VTM Spark virtual camera (compiled
// /target:winexe by build-run.ps1 into vendor\tools\vtm_spark_cam, next to the
// Unity Capture filters it registers).
//
// Windows' admin prompt shows the name and icon of the program it elevates.
// Elevating a .bat (cmd.exe) showed "Windows Command Processor" and the
// console icon; this exe carries the VTM Spark name and logo instead.
//
// Started without admin, it starts itself elevated (the one prompt) and hands
// back that run's exit code, so a plain CreateProcess caller can wait on it.
// Exit codes: 0 done, 1223 prompt declined, 2 a filter is missing,
// 3 registering failed.
using System;
using System.ComponentModel;
using System.Diagnostics;
using System.IO;
using System.Reflection;
using System.Security.Principal;

[assembly: AssemblyTitle("VTM Spark Camera Setup")]
[assembly: AssemblyDescription("Adds the VTM Spark virtual camera")]
[assembly: AssemblyCompany("VTM Studio")]
[assembly: AssemblyProduct("VTM Spark")]
[assembly: AssemblyCopyright("VTM Studio")]
[assembly: AssemblyVersion("1.0.0.0")]
[assembly: AssemblyFileVersion("1.0.0.0")]

internal static class Program
{
    // The name OBS, Discord and Zoom list the camera under.
    private const string DeviceName = "VTM Spark";
    private const string ElevatedFlag = "--elevated";
    private const int Declined = 1223; // ERROR_CANCELLED
    private const int Missing = 2;
    private const int Failed = 3;

    private static int Main(string[] args)
    {
        string exe = Assembly.GetExecutingAssembly().Location;
        string here = Path.GetDirectoryName(exe);
        // The flag stops a relaunch loop where elevation is not what it seems.
        if (!IsAdmin() && Array.IndexOf(args, ElevatedFlag) < 0)
        {
            return RunElevated(exe, here);
        }
        string dll32 = Path.Combine(here, "UnityCaptureFilter32.dll");
        string dll64 = Path.Combine(here, "UnityCaptureFilter64.dll");
        if (!File.Exists(dll32) || !File.Exists(dll64))
        {
            return Missing;
        }
        // Each filter with its own bitness of regsvr32 (SysWOW64 holds the
        // 32-bit one on 64-bit Windows).
        string sys = Environment.GetFolderPath(Environment.SpecialFolder.System);
        string sys32 = Environment.GetFolderPath(Environment.SpecialFolder.SystemX86);
        if (Register(Path.Combine(sys32, "regsvr32.exe"), dll32) != 0)
        {
            return Failed;
        }
        if (Register(Path.Combine(sys, "regsvr32.exe"), dll64) != 0)
        {
            return Failed;
        }
        return 0;
    }

    private static bool IsAdmin()
    {
        using (WindowsIdentity id = WindowsIdentity.GetCurrent())
        {
            return new WindowsPrincipal(id).IsInRole(WindowsBuiltInRole.Administrator);
        }
    }

    private static int RunElevated(string exe, string here)
    {
        ProcessStartInfo psi = new ProcessStartInfo
        {
            FileName = exe,
            Arguments = ElevatedFlag,
            WorkingDirectory = here,
            UseShellExecute = true,
            Verb = "runas",
        };
        try
        {
            using (Process p = Process.Start(psi))
            {
                if (p == null)
                {
                    return Failed;
                }
                p.WaitForExit();
                return p.ExitCode;
            }
        }
        catch (Win32Exception ex)
        {
            return ex.NativeErrorCode == Declined ? Declined : Failed;
        }
    }

    private static int Register(string regsvr32, string dll)
    {
        ProcessStartInfo psi = new ProcessStartInfo
        {
            FileName = regsvr32,
            Arguments = "/s \"" + dll + "\" \"/i:UnityCaptureName=" + DeviceName + "\"",
            WorkingDirectory = Path.GetDirectoryName(dll),
            UseShellExecute = false,
            CreateNoWindow = true,
        };
        using (Process p = Process.Start(psi))
        {
            p.WaitForExit();
            return p.ExitCode;
        }
    }
}
