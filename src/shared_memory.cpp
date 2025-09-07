#include <iostream>
#include <memory>
#include <Windows.h>
#include <climits>
#include <stdint.h>
#include <godot_cpp/variant/utility_functions.hpp>
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/classes/ref_counted.hpp>
#include <godot_cpp/classes/image.hpp>
#include <godot_cpp/variant/utility_functions.hpp>
#include <godot_cpp/variant/dictionary.hpp>
#include <godot_cpp/variant/string.hpp>
#include <godot_cpp/variant/packed_byte_array.hpp>
#include <godot_cpp/classes/project_settings.hpp>

#define SHARED_MEMORY_META           L"Local\\Godot_AI_Shared_Memory_Meta_ab5ec1ba-c1e2-46aa-9d61-7f93edebb97d"
#define SHARED_MEMORY_ACTION         L"Local\\Godot_AI_Shared_Memory_Action_ab5ec1ba-c1e2-46aa-9d61-7f93edebb97d"
#define SHARED_MEMORY_SCREENSHOT     L"Local\\Godot_AI_Shared_Memory_Screenshot_ab5ec1ba-c1e2-46aa-9d61-7f93edebb97d"
#define SHARED_MEMORY_OBSERVATION    L"Local\\Godot_AI_Shared_Memory_Observation_ab5ec1ba-c1e2-46aa-9d61-7f93edebb97d"

constexpr uint32_t MAX_SCREENSHOT_BUFFER_SIZE = 33177600;

struct alignas(64) Observation {
    float player_position_x;
    float player_position_y;
    float velocity_x;
    float velocity_y;
};

struct alignas(64) Meta {
    uint32_t screenshot_width;
    uint32_t screenshot_height;
    int32_t screenshot_format;
};

struct alignas(64) Action {
    int32_t action;
    int32_t reward;
    int8_t done;
    int8_t _padding[3];
};

#define SEMAPHORE_PYTHON             L"Local\\Godot_AI_Semaphore_Python_96c7e30b-4c86-4484-a18e-dacbffde8d72"
#define SEMAPHORE_GODOT              L"Local\\Godot_AI_Semaphore_Godot_96c7e30b-4c86-4484-a18e-dacbffde8d72"

#define MUTEX_APPLICATION            L"Local\\Godot_AI_Mutex_Application_e49b56cd-2851-4143-a856-b361b0ac1aa6"

struct HandleCloser {
    void operator()(HANDLE handle) {
        CloseHandle(handle);
    }
};

struct ViewUnMapper {
    void operator()(LPCVOID baseAddress) {
        UnmapViewOfFile(baseAddress);
    }
};

template <typename T>
class SharedMemory {
private:
    SIZE_T m_FileOffset;
    SYSTEM_INFO m_SystemInfo;
    MEMORY_BASIC_INFORMATION m_MemoryInfo;
    std::unique_ptr<void, HandleCloser> m_Handle;
    std::unique_ptr<T, ViewUnMapper> m_Ptr;

public:
    const wchar_t* const Name;
    const SIZE_T Size;

    SharedMemory(const SIZE_T size, const wchar_t* const name) 
    : Name{name}, Size{size}, m_Handle{nullptr}, m_Ptr{nullptr}
    {
        GetSystemInfo(&m_SystemInfo);
    }
    ~SharedMemory() = default;
    SharedMemory(const SharedMemory&) = delete;
    SharedMemory& operator=(const SharedMemory&) = delete;
    SharedMemory(SharedMemory&&) = default;
    SharedMemory& operator=(SharedMemory&&) = default;

    T* Get() { return m_Ptr.get(); }

    HANDLE GetHandle() { return m_Handle.get(); }

    T& operator*() {
        #ifdef DEBUG_ENABLED
        if (!m_Ptr.get()) throw std::runtime_error("Dereferencing null pointer");
        #endif
        return *m_Ptr.get();
    }
    
    const T& operator*() const {
        #ifdef DEBUG_ENABLED
        if (!m_Ptr.get()) throw std::runtime_error("Dereferencing null pointer");
        #endif
        return *m_Ptr.get();
    }
    
    T* operator->() {
        #ifdef DEBUG_ENABLED
        if (!m_Ptr.get()) throw std::runtime_error("Accessing null pointer");
        #endif
        return m_Ptr.get();
    }
    
    const T* operator->() const {
        #ifdef DEBUG_ENABLED
        if (!m_Ptr.get()) throw std::runtime_error("Accessing null pointer");
        #endif
        return m_Ptr.get();
    }

    bool IsNull() const { return m_Ptr == nullptr; }
    
    bool IsValid() const { return m_Ptr != nullptr; }
    
    explicit operator bool() const { return m_Ptr != nullptr; }

    DWORD High(const SIZE_T number) const { return static_cast<DWORD>(number >> sizeof(DWORD)*CHAR_BIT); }
    
    DWORD Low (const SIZE_T Number) const { return static_cast<DWORD>(static_cast<DWORD>(-1) & Number); }

    SIZE_T GetTotalSizeCreated() { return (m_Handle) ? (Size/m_SystemInfo.dwPageSize +1)*m_SystemInfo.dwPageSize : -1; }

    SIZE_T GetActualSizeMapped() { return (m_Ptr) ? m_MemoryInfo.RegionSize : -1; }

    SIZE_T GetFileOffset() { return (m_Ptr) ? m_FileOffset : -1; }

    uint32_t Create() {
        HANDLE handle = CreateFileMappingW(INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE, High(Size), Low(Size), Name);
        if (handle) {
            m_Handle.reset(handle);
            if (GetLastError() == ERROR_ALREADY_EXISTS) return ERROR_ALREADY_EXISTS;
            else return 0;
        }
        return GetLastError();
    }

    uint32_t Map(const SIZE_T granularityMultiplier = 0, const SIZE_T  dwNumberOfBytesToMap = 0) {
        m_FileOffset = m_SystemInfo.dwAllocationGranularity * granularityMultiplier;
        T* ptr = reinterpret_cast<T*>(MapViewOfFile(m_Handle.get(), FILE_MAP_ALL_ACCESS, High(m_FileOffset), Low(m_FileOffset), dwNumberOfBytesToMap));
        if (ptr) {
            m_Ptr.reset(ptr);
            #ifdef DEBUG_ENABLED
            godot::UtilityFunctions::print("Connected: ", Name, " | size: ", godot::String::num_uint64(High(Size)), " " , godot::String::num_uint64(Low(Size)));
            #endif
            if (!VirtualQuery(ptr, &m_MemoryInfo, sizeof(m_MemoryInfo))) return GetLastError();
            else return 0;
        }
        return GetLastError();
    }

    void Close() {
        m_Handle.reset();
        m_Ptr.reset();
    }

};


class Semaphore {
private:
    std::unique_ptr<void, HandleCloser> m_Handle;

public:
    const wchar_t* const Name;
    const LONG InitialCount;
    const LONG MaxCount;
    LONG PreviousCount;

    Semaphore(const LONG initialCount, const LONG maxCount, const wchar_t* const name) 
    : Name{name}, InitialCount{initialCount}, MaxCount{maxCount}, PreviousCount{-1}, m_Handle{nullptr}
    {}

    ~Semaphore() = default;
    Semaphore(const Semaphore&) = delete;
    Semaphore& operator=(const Semaphore&) = delete;
    Semaphore(Semaphore&&) = default;
    Semaphore& operator=(Semaphore&&) = default;

    HANDLE GetHandle() { return m_Handle.get(); }

    bool IsNull() const { return m_Handle == nullptr; }
    
    bool IsValid() const { return m_Handle != nullptr; }
    
    explicit operator bool() const { return m_Handle != nullptr; }

    uint32_t Create() {
        HANDLE handle = CreateSemaphoreW(NULL, InitialCount, MaxCount, Name);
        if (handle) {
            m_Handle.reset(handle);
            #ifdef DEBUG_ENABLED
            godot::UtilityFunctions::print("Connected: ", Name);
            #endif
            if (GetLastError() == ERROR_ALREADY_EXISTS) return ERROR_ALREADY_EXISTS;
            else return 0;
        }
        return GetLastError();
    }

    uint32_t Wait(DWORD dwMilliseconds) {
        DWORD result = WaitForSingleObject(m_Handle.get(), dwMilliseconds);
        return result;
    }

    uint32_t Go(LONG lReleaseCount = 1) {
        if (ReleaseSemaphore(m_Handle.get(), lReleaseCount, NULL)) return 0;
        else return GetLastError();
    }

    uint32_t SaveAndGo(LONG lReleaseCount = 1) {
        if (ReleaseSemaphore(m_Handle.get(), lReleaseCount, &PreviousCount)) return 0;
        else return GetLastError();
    }

    void Close() {
        m_Handle.reset();
    }
};


class Mutex {
private:
    std::unique_ptr<void, HandleCloser> m_Handle;

public:
    const wchar_t* const Name;
    const bool IsInitialOwner;

    Mutex(const bool isInitialOwner, const wchar_t* const name) 
    : Name{name}, IsInitialOwner{isInitialOwner}, m_Handle{nullptr}
    {}

    ~Mutex() = default;
    Mutex(const Mutex&) = delete;
    Mutex& operator=(const Mutex&) = delete;
    Mutex(Mutex&&) = default;
    Mutex& operator=(Mutex&&) = default;

    HANDLE GetHandle() { return m_Handle.get(); }

    bool IsNull() const { return m_Handle == nullptr; }
    
    bool IsValid() const { return m_Handle != nullptr; }
    
    explicit operator bool() const { return m_Handle != nullptr; }

    uint32_t Create() {
        HANDLE handle = CreateMutexW(NULL, IsInitialOwner, Name);
        if (handle) {
            m_Handle.reset(handle);
            #ifdef DEBUG_ENABLED
            godot::UtilityFunctions::print("Connected: ", Name);
            #endif
            if (GetLastError() == ERROR_ALREADY_EXISTS) return ERROR_ALREADY_EXISTS;
            else return 0;
        }
        return GetLastError();
    }

    uint32_t Wait(DWORD dwMilliseconds) {
        DWORD result = WaitForSingleObject(m_Handle.get(), dwMilliseconds);
        return result;
    }

    uint32_t Go() {
        if (ReleaseMutex(m_Handle.get())) return 0;
        else return GetLastError();
    }

    void Close() {
        m_Handle.reset();
    }
};

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/classes/ref_counted.hpp>
#include <godot_cpp/classes/image.hpp>
#include <godot_cpp/variant/utility_functions.hpp>
#include <godot_cpp/variant/dictionary.hpp>
#include <godot_cpp/variant/string.hpp>
#include <godot_cpp/variant/packed_byte_array.hpp>
#include <godot_cpp/classes/project_settings.hpp>
#include <algorithm>

namespace godot {

class SharedMemoryLink : public RefCounted {
    GDCLASS(SharedMemoryLink, RefCounted);

private:
    SharedMemory     <Meta>       m_Meta{sizeof(Meta), SHARED_MEMORY_META};
    SharedMemory    <Action>      m_Action{sizeof(Action), SHARED_MEMORY_ACTION};
    SharedMemory  <Observation>   m_Observation{sizeof(Observation), SHARED_MEMORY_OBSERVATION};
    SharedMemory <unsigned char>  m_Screenshot{MAX_SCREENSHOT_BUFFER_SIZE, SHARED_MEMORY_SCREENSHOT};

    Semaphore m_PythonSemaphore {0, 1, SEMAPHORE_PYTHON};
    Semaphore m_GodotSemaphore  {1, 1, SEMAPHORE_GODOT};

    Mutex m_ApplicationMutex    {false, MUTEX_APPLICATION};

    const DWORD TIMEOUT_MS = 60000; // 1 minute timeout
    std::mutex m_CacheMutex;

    bool m_IsConnected = false;
    bool m_IsFirstInstance = false;

protected:
    static void _bind_methods() {
        ClassDB::bind_method(D_METHOD("connect"), &SharedMemoryLink::Connect);
        ClassDB::bind_method(D_METHOD("disconnect"), &SharedMemoryLink::Disconnect);
        ClassDB::bind_method(D_METHOD("read_action"), &SharedMemoryLink::ReadAction);
        ClassDB::bind_method(D_METHOD("is_connected"), &SharedMemoryLink::IsConnected);
        ClassDB::bind_method(D_METHOD("is_first_instance"), &SharedMemoryLink::IsFirstInstance);
        ClassDB::bind_method(D_METHOD("send_meta", "screenshot"), &SharedMemoryLink::SendMeta);
        ClassDB::bind_method(D_METHOD("send_step_dictionary", "observation", "screenshot", "reward", "done"), &SharedMemoryLink::SendStepDictionary);
        ClassDB::bind_method(D_METHOD("send_step", "player_position_x", "player_position_y", "player_velocity_x", "player_velocity_y", "screenshot", "reward", "done"), &SharedMemoryLink::SendStep);
    }

public:
    SharedMemoryLink() = default;
    ~SharedMemoryLink() { Disconnect(); }

    bool Connect() {
        #ifdef DEBUG_ENABLED
        UtilityFunctions::print("Connecting SharedMemoryLink...");
        #endif
        if (m_IsConnected) {
            UtilityFunctions::print("Already connected to shared memory.");
            return true;
        }

        try {
            std::array<bool, 4> alreadyExists{false, false, false, false};
            uint32_t result;

            result = m_Observation.Create();
            alreadyExists[0] = (result == ERROR_ALREADY_EXISTS);
            if (result != 0 && result != ERROR_ALREADY_EXISTS) {
                UtilityFunctions::printerr("Failed to create observation shared memory. Error: ", (int)result);
                return false;
            }

            result = m_Screenshot.Create();
            alreadyExists[1] = (result == ERROR_ALREADY_EXISTS);
            if (result != 0 && result != ERROR_ALREADY_EXISTS) {
                UtilityFunctions::printerr("Failed to create screenshot shared memory. Error: ", (int)result);
                return false;
            }

            result = m_Meta.Create();
            alreadyExists[2] = (result == ERROR_ALREADY_EXISTS);
            if (result != 0 && result != ERROR_ALREADY_EXISTS) {
                UtilityFunctions::printerr("Failed to create meta shared memory. Error: ", (int)result);
                return false;
            }

            result = m_Action.Create();
            alreadyExists[3] = (result == ERROR_ALREADY_EXISTS);
            if (result != 0 && result != ERROR_ALREADY_EXISTS) {
                UtilityFunctions::printerr("Failed to create action shared memory. Error: ", (int)result);
                return false;
            }

            if (m_Observation.Map()) {
                UtilityFunctions::printerr("Failed to map observation shared memory.");
                return false;
            }

            if (m_Screenshot.Map()) {
                UtilityFunctions::printerr("Failed to map screenshot shared memory.");
                return false;
            }

            if (m_Meta.Map()) {
                UtilityFunctions::printerr("Failed to map meta shared memory.");
                return false;
            }

            if (m_Action.Map()) {
                UtilityFunctions::printerr("Failed to map action shared memory.");
                return false;
            }

            bool allAlreadyExist = std::all_of(alreadyExists.begin(), alreadyExists.end(), [](bool val) { 
                return val; 
            });

            result = m_ApplicationMutex.Create();
            if (result != 0 && result != ERROR_ALREADY_EXISTS) {
                UtilityFunctions::printerr("Failed to create application mutex. Error: ", (int)result);
                return false;
            }

            result = m_ApplicationMutex.Wait(0);

            if (result == WAIT_ABANDONED) {
                #ifdef DEBUG_ENABLED
                UtilityFunctions::print("Detected abandoned mutex. Taking ownership and re-initializing.");
                #endif
                m_IsFirstInstance = true;
                memset(m_Observation.Get(), 0, m_Observation.Size);
                memset(m_Meta.Get(),        0, m_Meta.Size);
                memset(m_Screenshot.Get(),  0, m_Screenshot.Size);
                memset(m_Action.Get(),      0, m_Action.Size);

            } else if (allAlreadyExist && result == WAIT_TIMEOUT) {
                #ifdef DEBUG_ENABLED
                UtilityFunctions::print("Another instance is already running. Bringing it to foreground and exiting.");
                #endif
                m_IsFirstInstance = false;

                String app_name = ProjectSettings::get_singleton()->get_setting("application/config/name");
                std::wstring windowTitle = std::wstring(reinterpret_cast<const wchar_t*>(app_name.utf16().get_data()));
                HWND hWnd = FindWindowW(NULL, windowTitle.c_str());
                if (hWnd) {
                    if (IsIconic(hWnd)) {
                        ShowWindow(hWnd, SW_RESTORE);
                    }
                    SetForegroundWindow(hWnd);
                }
                return false;

            } else if (result == WAIT_OBJECT_0) {
                #ifdef DEBUG_ENABLED
                UtilityFunctions::print("This is the first instance.");
                #endif
                m_IsFirstInstance = true;

            } else {
                UtilityFunctions::printerr("Unexpected mutex wait result: ", (int)result);
                return false;
            }

            result = m_GodotSemaphore.Create();
            if (result != 0 && result != ERROR_ALREADY_EXISTS) {
                UtilityFunctions::printerr("Failed to create Godot semaphore. Error: ", (int)result);
                return false;
            }

            result = m_PythonSemaphore.Create();
            if (result != 0 && result != ERROR_ALREADY_EXISTS) {
                UtilityFunctions::printerr("Failed to create Python semaphore. Error: ", (int)result);
                return false;
            }

            m_IsConnected = true;
            #ifdef DEBUG_ENABLED
            UtilityFunctions::print("SharedMemoryLink connection established successfully.");
            #endif
            return true;

        } catch (const std::exception& ex) {
            UtilityFunctions::printerr("SharedMemoryLink connection failed: ", ex.what());
            Disconnect();
            return false;
        }
    }

    void Disconnect() {
        if (!m_IsConnected) return;

        #ifdef DEBUG_ENABLED
        UtilityFunctions::print("Disconnecting SharedMemoryLink...");
        #endif
        m_IsConnected = false;

        m_Meta.Close();
        m_Action.Close();
        m_Screenshot.Close();
        m_Observation.Close();

        m_ApplicationMutex.Close();
        m_PythonSemaphore.Close();
        m_GodotSemaphore.Close();

        #ifdef DEBUG_ENABLED
        UtilityFunctions::print("SharedMemoryLink disconnected successfully.");
        #endif
    }

    bool SendMeta(const Ref<Image>& screenshot) {
        if (!m_IsConnected || !m_IsFirstInstance) {
            UtilityFunctions::printerr("Not connected or not the first instance.");
            return false;
        }
        uint32_t result;

        result = m_GodotSemaphore.Wait(TIMEOUT_MS);
        if (result != WAIT_OBJECT_0) {
            UtilityFunctions::printerr("Timeout waiting for Godot semaphore. Result: ", (int)result);
            return false;
        }

        try {
            std::lock_guard<std::mutex> lock(m_CacheMutex);

            m_Meta->screenshot_width  = screenshot->get_width();
            m_Meta->screenshot_height = screenshot->get_height();
            m_Meta->screenshot_format = screenshot->get_format();

            result = m_PythonSemaphore.Go();
            if (result != 0) {
                UtilityFunctions::printerr("Failed to signal Python semaphore. Error: ", (int)result);
                return false;
            }

            return true;

        } catch (const std::exception& ex) {
            UtilityFunctions::printerr("Error in SendStep: ", ex.what());
            m_PythonSemaphore.Go();
            return false;
        }
    }

    bool SendStepDictionary(const Dictionary& observation, const Ref<Image>& screenshot, int reward, bool done) {
        if (!m_IsConnected || !m_IsFirstInstance) {
            UtilityFunctions::printerr("Not connected or not the first instance.");
            return false;
        }
        uint32_t result;

        result = m_GodotSemaphore.Wait(TIMEOUT_MS);
        if (result != WAIT_OBJECT_0) {
            UtilityFunctions::printerr("Timeout waiting for Godot semaphore. Result: ", (int)result);
            return false;
        }

        try {
            std::lock_guard<std::mutex> lock(m_CacheMutex);

            Observation* obsData = m_Observation.Get();

            obsData->player_position_x = observation["Player_position_x"];
            obsData->player_position_y = observation["Player_position_y"];
            obsData->player_position_x = observation["Player_velocity_x"];
            obsData->player_position_y = observation["Player_velocity_y"];


            PackedByteArray imgData = screenshot->get_data();
            uint32_t imgSize = std::min(static_cast<uint32_t>(imgData.size()), MAX_SCREENSHOT_BUFFER_SIZE);
                
            memcpy(m_Screenshot.Get(), imgData.ptr(), imgSize);
                

            m_Action->reward = reward;
            m_Action->done = done ? 1 : 0;
            m_Action->action = -1;

            result = m_PythonSemaphore.Go();
            if (result != 0) {
                UtilityFunctions::printerr("Failed to signal Python semaphore. Error: ", (int)result);
                return false;
            }

            return true;

        } catch (const std::exception& ex) {
            UtilityFunctions::printerr("Error in SendStep: ", ex.what());
            m_PythonSemaphore.Go();
            return false;
        }
    }

    bool SendStep(float player_position_x, float player_position_y, float player_velocity_x, float player_velocity_y,  const Ref<Image>& screenshot, int reward, bool done) {
        if (!m_IsConnected || !m_IsFirstInstance) {
            UtilityFunctions::printerr("Not connected or not the first instance.");
            return false;
        }
        uint32_t result;

        result = m_GodotSemaphore.Wait(TIMEOUT_MS);
        if (result != WAIT_OBJECT_0) {
            UtilityFunctions::printerr("Timeout waiting for Godot semaphore. Result: ", (int)result);
            return false;
        }

        try {
            std::lock_guard<std::mutex> lock(m_CacheMutex);

            Observation* obsData = m_Observation.Get();

            obsData->player_position_x = player_position_x;
            obsData->player_position_y = player_position_y;
            obsData->player_position_x = player_position_x;
            obsData->player_position_y = player_position_y;


            PackedByteArray imgData = screenshot->get_data();
            uint32_t imgSize = std::min(static_cast<uint32_t>(imgData.size()), MAX_SCREENSHOT_BUFFER_SIZE);
                
            memcpy(m_Screenshot.Get(), imgData.ptr(), imgSize);
                

            m_Action->reward = reward;
            m_Action->done = done ? 1 : 0;

            result = m_PythonSemaphore.Go();
            if (result != 0) {
                UtilityFunctions::printerr("Failed to signal Python semaphore. Error: ", (int)result);
                return false;
            }

            return true;

        } catch (const std::exception& ex) {
            UtilityFunctions::printerr("Error in SendStep: ", ex.what());
            m_PythonSemaphore.Go();
            return false;
        }
    }

    int ReadAction() {
        if (!m_IsConnected || !m_IsFirstInstance) {
            UtilityFunctions::printerr("Not connected or not the first instance.");
            return -1;
        }

        std::lock_guard<std::mutex> lock(m_CacheMutex);
        UtilityFunctions::print("Action: ",m_Action->action);
        return m_Action->action;
    }

    bool IsConnected() const {
        return m_IsConnected;
    }

    bool IsFirstInstance() const {
        return m_IsFirstInstance;
    }
};

} // namespace godot

#include <gdextension_interface.h>
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/defs.hpp>
#include <godot_cpp/godot.hpp>

using namespace godot;

void initialize_shml(ModuleInitializationLevel p_level) {
    if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) {
        return;
    }
    ClassDB::register_class<SharedMemoryLink>();
}

void uninitialize_shml(ModuleInitializationLevel p_level) {
    if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) {
        return;
    }
}


extern "C" {
    GDExtensionBool GDE_EXPORT shml_init(
        GDExtensionInterfaceGetProcAddress p_get_proc_address,
        const GDExtensionClassLibraryPtr p_library,
        GDExtensionInitialization* r_initialization
    ){
        GDExtensionBinding::InitObject init_obj(p_get_proc_address, p_library, r_initialization);
        init_obj.register_initializer(initialize_shml);
        init_obj.register_terminator(uninitialize_shml);
        init_obj.set_minimum_library_initialization_level(MODULE_INITIALIZATION_LEVEL_SCENE);

        return init_obj.init();
    }
}