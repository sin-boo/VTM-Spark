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
//
// With --uninstall (uninstall.bat) it unregisters the camera and removes the
// Program Files copy instead; copies still loaded go at the next restart.
//
// It registers a copy under Program Files, not the filters next to it. Every
// program that lists webcams (browsers, OBS, Discord) loads the registered DLL
// and holds it open; registered from vendor\ that locked the app folder, which
// then could not be deleted or replaced. A copy already held that way moves out
// of its app folder (see Release), so the folder deletes without a restart.
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Diagnostics;
using System.IO;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Security.Principal;
using Microsoft.Win32;

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
    private const string UninstallFlag = "--uninstall";
    private const int Declined = 1223; // ERROR_CANCELLED
    private const int Missing = 2;
    private const int Failed = 3;
    // DirectShow "Video Input Device" category: where Windows lists webcams.
    private const string VideoInput = @"SOFTWARE\Classes\CLSID\{860BB310-5D01-11d0-BD3B-00A0C911CE86}\Instance";

    private static int Main(string[] args)
    {
        string exe = Assembly.GetExecutingAssembly().Location;
        string here = Path.GetDirectoryName(exe);
        bool uninstall = Array.IndexOf(args, UninstallFlag) >= 0;
        // The flag stops a relaunch loop where elevation is not what it seems.
        if (!IsAdmin() && Array.IndexOf(args, ElevatedFlag) < 0)
        {
            return RunElevated(exe, here, uninstall ? ElevatedFlag + " " + UninstallFlag : ElevatedFlag);
        }
        if (uninstall)
        {
            return Uninstall(here);
        }
        string dll32 = Path.Combine(here, "UnityCaptureFilter32.dll");
        string dll64 = Path.Combine(here, "UnityCaptureFilter64.dll");
        if (!File.Exists(dll32) || !File.Exists(dll64))
        {
            return Missing;
        }
        string dir;
        // Copies programs may still hold: what the camera was registered from
        // until now, and the filters next to this exe.
        HashSet<string> before = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        try
        {
            dir = InstallDir();
            Directory.CreateDirectory(dir);
            // Earlier versions moved copies aside into dir itself.
            ClearOld(dir);
            ClearOld(AsideDir(dir));
            ClearLegacyLeftovers();
            foreach (string old in RegisteredFilters())
            {
                before.Add(old);
            }
            before.Add(dll32);
            before.Add(dll64);
            dll32 = Place(dll32, dir);
            dll64 = Place(dll64, dir);
        }
        catch (Exception)
        {
            return Failed;
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
        foreach (string old in before)
        {
            Release(old, dir);
        }
        return 0;
    }

    // Unregisters every copy the camera is registered from (and the bundled and
    // Program Files ones), then removes Program Files\VTM Spark.
    private static int Uninstall(string here)
    {
        string dir = InstallDir();
        HashSet<string> dlls = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        foreach (string dll in RegisteredFilters())
        {
            dlls.Add(dll);
        }
        foreach (string name in new[] { "UnityCaptureFilter32.dll", "UnityCaptureFilter64.dll" })
        {
            dlls.Add(Path.Combine(dir, name));
            dlls.Add(Path.Combine(here, name));
        }
        string sys = Environment.GetFolderPath(Environment.SpecialFolder.System);
        string sys32 = Environment.GetFolderPath(Environment.SpecialFolder.SystemX86);
        foreach (string dll in dlls)
        {
            if (!File.Exists(dll))
            {
                continue;
            }
            bool x86 = Path.GetFileNameWithoutExtension(dll).EndsWith("32", StringComparison.OrdinalIgnoreCase);
            Unregister(Path.Combine(x86 ? sys32 : sys, "regsvr32.exe"), dll);
        }
        ClearLegacyLeftovers();
        RemoveTree(Path.GetDirectoryName(dir));
        return RegisteredFilters().Count == 0 ? 0 : Failed;
    }

    // Deletes what it can; what a program still has loaded goes at the next restart.
    private static void RemoveTree(string path)
    {
        if (!Directory.Exists(path))
        {
            return;
        }
        foreach (string file in Directory.GetFiles(path, "*", SearchOption.AllDirectories))
        {
            try
            {
                File.Delete(file);
            }
            catch (Exception)
            {
                MoveFileEx(file, null, MoveFileDelayUntilReboot);
            }
        }
        string[] dirs = Directory.GetDirectories(path, "*", SearchOption.AllDirectories);
        Array.Sort(dirs, (a, b) => b.Length.CompareTo(a.Length));
        foreach (string sub in dirs)
        {
            try { Directory.Delete(sub); } catch (Exception) { MoveFileEx(sub, null, MoveFileDelayUntilReboot); }
        }
        try { Directory.Delete(path); } catch (Exception) { MoveFileEx(path, null, MoveFileDelayUntilReboot); }
    }

    // Filter DLLs the VTM Spark camera is registered from, 64- and 32-bit
    // (backend/vcam_device.py registered_filters reads the same keys).
    private static List<string> RegisteredFilters()
    {
        List<string> found = new List<string>();
        foreach (RegistryView view in new[] { RegistryView.Registry64, RegistryView.Registry32 })
        {
            using (RegistryKey hklm = RegistryKey.OpenBaseKey(RegistryHive.LocalMachine, view))
            using (RegistryKey devices = hklm.OpenSubKey(VideoInput))
            {
                if (devices == null)
                {
                    continue;
                }
                foreach (string sub in devices.GetSubKeyNames())
                {
                    using (RegistryKey key = devices.OpenSubKey(sub))
                    {
                        if (key == null || !DeviceName.Equals(key.GetValue("FriendlyName") as string))
                        {
                            continue;
                        }
                        string clsid = key.GetValue("CLSID") as string;
                        if (string.IsNullOrEmpty(clsid))
                        {
                            continue;
                        }
                        using (RegistryKey server = hklm.OpenSubKey(@"SOFTWARE\Classes\CLSID\" + clsid + @"\InprocServer32"))
                        {
                            string dll = server == null ? null : server.GetValue("") as string;
                            if (!string.IsNullOrEmpty(dll))
                            {
                                found.Add(dll.Trim('"'));
                            }
                        }
                    }
                }
            }
        }
        return found;
    }

    // A copy outside dir that some program still has loaded keeps its app
    // folder from being deleted until that program closes. A loaded DLL cannot
    // be deleted but can be moved within its drive: it moves out of the app
    // folder (gone at the next restart) and a copy nothing holds takes its
    // place, so the app still finds its filters there.
    private static void Release(string path, string dir)
    {
        try
        {
            string fresh = Path.Combine(dir, Path.GetFileName(path));
            if (!File.Exists(path) || !File.Exists(fresh) || SameFolder(Path.GetDirectoryName(path), dir))
            {
                return;
            }
            try
            {
                File.Delete(path);
            }
            catch (UnauthorizedAccessException)
            {
                if (!MoveAside(path, dir))
                {
                    return;
                }
            }
            catch (IOException)
            {
                if (!MoveAside(path, dir))
                {
                    return;
                }
            }
            File.Copy(fresh, path);
        }
        catch (Exception)
        {
            // The camera is registered either way; the old folder only stays
            // locked until the programs holding it close.
        }
    }

    // Into dir\old, emptied at the next restart. Only when path is on dir's
    // drive (a loaded DLL cannot move across drives); otherwise it stays put
    // and its folder stays locked until the programs holding it close.
    private static bool MoveAside(string path, string dir)
    {
        string root = Path.GetPathRoot(Path.GetFullPath(path));
        if (!string.Equals(root, Path.GetPathRoot(dir), StringComparison.OrdinalIgnoreCase))
        {
            return false;
        }
        string aside = AsideDir(dir);
        Directory.CreateDirectory(aside);
        string moved = Path.Combine(aside, Path.GetFileName(path) + "." + DateTime.Now.Ticks + ".old");
        File.Move(path, moved);
        MoveFileEx(moved, null, MoveFileDelayUntilReboot);
        // After the file it holds; removes only an empty folder.
        MoveFileEx(aside, null, MoveFileDelayUntilReboot);
        return true;
    }

    // Where copies still loaded wait for the next restart: a plain folder
    // inside dir, so on dir's drive.
    private static string AsideDir(string dir)
    {
        return Path.Combine(dir, "old");
    }

    // Earlier versions moved copies held on another drive into a hidden
    // "VTM Spark leftovers" folder at that drive's root. Empties and removes
    // any still there; nothing creates it any more.
    private static void ClearLegacyLeftovers()
    {
        foreach (DriveInfo drive in DriveInfo.GetDrives())
        {
            try
            {
                if (drive.DriveType != DriveType.Fixed || !drive.IsReady)
                {
                    continue;
                }
                string legacy = Path.Combine(drive.RootDirectory.FullName, "VTM Spark leftovers");
                if (!Directory.Exists(legacy))
                {
                    continue;
                }
                ClearOld(legacy);
                Directory.Delete(legacy);
            }
            catch (Exception)
            {
                // Still holds a copy some program has loaded; that run already
                // set it to go at the next restart.
            }
        }
    }

    private static bool SameFolder(string a, string b)
    {
        return string.Equals(
            Path.GetFullPath(a).TrimEnd('\\'),
            Path.GetFullPath(b).TrimEnd('\\'),
            StringComparison.OrdinalIgnoreCase);
    }

    // backend/vcam_device.py installed_dir() reads the same place. ProgramW6432
    // is the 64-bit Program Files even when this runs as a 32-bit process.
    private static string InstallDir()
    {
        string root = Environment.GetEnvironmentVariable("ProgramW6432");
        if (string.IsNullOrEmpty(root))
        {
            root = Environment.GetFolderPath(Environment.SpecialFolder.ProgramFiles);
        }
        return Path.Combine(Path.Combine(root, "VTM Spark"), "Camera");
    }

    // Copies a filter into dir and returns the copy. A copy some program still
    // has loaded cannot be overwritten or deleted, only renamed: it moves aside
    // and goes at the next restart.
    private static string Place(string src, string dir)
    {
        string dest = Path.Combine(dir, Path.GetFileName(src));
        if (File.Exists(dest))
        {
            if (SameBytes(src, dest))
            {
                return dest;
            }
            MoveAside(dest, dir);
        }
        File.Copy(src, dest);
        return dest;
    }

    // Copies moved aside by an earlier run, once nothing holds them any more.
    private static void ClearOld(string dir)
    {
        if (!Directory.Exists(dir))
        {
            return;
        }
        foreach (string old in Directory.GetFiles(dir, "*.old"))
        {
            try
            {
                File.Delete(old);
            }
            catch (Exception)
            {
            }
        }
    }

    private static bool SameBytes(string a, string b)
    {
        byte[] x = File.ReadAllBytes(a);
        byte[] y = File.ReadAllBytes(b);
        if (x.Length != y.Length)
        {
            return false;
        }
        for (int i = 0; i < x.Length; i++)
        {
            if (x[i] != y[i])
            {
                return false;
            }
        }
        return true;
    }

    private const int MoveFileDelayUntilReboot = 0x4;

    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    private static extern bool MoveFileEx(string existing, string replacement, int flags);

    private static bool IsAdmin()
    {
        using (WindowsIdentity id = WindowsIdentity.GetCurrent())
        {
            return new WindowsPrincipal(id).IsInRole(WindowsBuiltInRole.Administrator);
        }
    }

    private static int RunElevated(string exe, string here, string args)
    {
        ProcessStartInfo psi = new ProcessStartInfo
        {
            FileName = exe,
            Arguments = args,
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

    private static int Unregister(string regsvr32, string dll)
    {
        ProcessStartInfo psi = new ProcessStartInfo
        {
            FileName = regsvr32,
            Arguments = "/u /s \"" + dll + "\" \"/i:UnityCaptureName=" + DeviceName + "\"",
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
