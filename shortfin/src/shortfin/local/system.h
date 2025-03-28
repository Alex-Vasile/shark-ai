// Copyright 2024 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef SHORTFIN_LOCAL_SYSTEM_H
#define SHORTFIN_LOCAL_SYSTEM_H

#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "shortfin/local/device.h"
#include "shortfin/local/messaging.h"
#include "shortfin/local/worker.h"
#include "shortfin/support/api.h"
#include "shortfin/support/blocking_executor.h"
#include "shortfin/support/config.h"
#include "shortfin/support/iree_concurrency.h"
#include "shortfin/support/iree_helpers.h"
#include "shortfin/support/stl_extras.h"

namespace shortfin::local {

namespace detail {
class BaseProcess;
}  // namespace detail

class Fiber;
class System;
class SystemBuilder;

// Encapsulates resources attached to the local system. In most applications,
// there will be one of these, and it is used to keep long lived access to
// physical devices, connections, and other long lived resources which need
// to be available across the application lifetime.
//
// One does not generally construct a System by hand, instead relying
// on some form of factory that constructs one to suit both the system being
// executed on and any preferences on which resources should be accessible.
//
// Ownership
// ---------
// There are three levels of ownership, all rooted on the System:
//   1. System: The System class, all drivers, devices, workers, and executors.
//      There will only ever be one (or a small number if doing something multi
//      tenant), and all owning references to the System are via
//      `std::shared_ptr<System>`. Every object in the system must either be
//      a managed child of the system or own a system reference.
//   2. Fiber: Binds any number of devices to a coherent schedule, rooted on
//      a Worker. Scopes are independent of the system and there are generally
//      as many as needed logical concurrency in the application. Each fiber
//      holds a system reference by way of a `std::shared_ptr<System>`. These
//      are still heavy-weight objects mostly created at initialization time
//      and are therefore managed held as a `std::shared_ptr<Fiber>` by anything
//      that depends on them.
//   3. TimelineResource: Any resource in the system (i.e. buffer,
//      synchronization, object, etc) will hold a unique TimelineResource. These
//      are light-weight objects managed via intrusive reference counting by
//      their contained `TimelineResource::Ref` class. Each `TimelineResource`
//      maintains a `std::shared_ptr<Fiber>` back reference to its owning
//      fiber.
//
// Leaf objects can have any lifetime that they wish, so long as they maintain
// an appropriate ownership reference into the System hierarchy above. This
// includes any application managed objects like arrays, storage, processes,
// messages, queues, etc.
//
// Lifetime debug logging can be enabled via compiler defines:
//   SHORTFIN_LOG_LIFETIMES=1 : Enables constructor/destructor and this pointer
//     logging for the primary objects in the system hierarchy.
//   SHORTFIN_IREE_LOG_RC=1 : Enables the application view of IREE object
//     reference counting, showing steal/retain/release and the number of
//     references the application holds for each object. Also will log any
//     outstanding references when the System is deallocated.
class SHORTFIN_API System : public std::enable_shared_from_this<System> {
 public:
  System(iree_allocator_t host_allocator);
  System(const System &) = delete;
  ~System();

  // One shot creation factory that is the equivalent of:
  //   SystemBuilder::ForSystem(
  //     host_allocator, system_type,
  //     std::move(config_options))->CreateSystem()
  // Undef validation will be done on the config options prior to returning.
  static std::shared_ptr<System> Create(iree_allocator_t host_allocator,
                                        std::string_view system_type,
                                        ConfigOptions config_options = {});

  // Explicit shutdown (vs in destructor) is encouraged.
  void Shutdown();

  // Get a shared pointer from the instance.
  std::shared_ptr<System> shared_ptr() { return shared_from_this(); }

  // Access to underlying IREE API objects.
  iree_allocator_t host_allocator() { return host_allocator_; }
  iree_vm_instance_t *vm_instance() { return vm_instance_.get(); }

  // Topology access.
  std::span<const Node> nodes() { return {nodes_}; }
  std::span<Device *const> devices() { return {devices_}; }
  const std::unordered_map<std::string_view, Device *> named_devices() {
    return named_devices_;
  }
  Device *FindDeviceByName(std::string_view name);

  // Queue access.
  QueuePtr CreateQueue(Queue::Options options);
  QueuePtr CreateQueue() { return CreateQueue(Queue::Options()); }
  QueuePtr named_queue(std::string_view name);

  // Access the system wide blocking executor thread pool. This can be used
  // to execute thunks that can block on a dedicated thread and is needed
  // to bridge APIs that cannot be used in a non-blocking context.
  BlockingExecutor &blocking_executor() { return blocking_executor_; }

  // Scopes.
  // Creates a new Fiber bound to this System (it will internally
  // hold a reference to this instance). All devices in system order will be
  // added to the fiber.
  std::shared_ptr<Fiber> CreateFiber(Worker &worker,
                                     std::span<Device *const> devices);

  // Creates and starts a worker (if it is configured to run in a thread).
  Worker &CreateWorker(Worker::Options options);

  // Accesses the initialization worker that is intended to be run on the main
  // or adopted thread to perform any async interactions with the system.
  // Internally, this worker is called "__init__". It will be created on
  // demand if it does not yet exist.
  Worker &init_worker();

  // Adds a worker initializer which will be called when each worker starts.
  // This can only be called before any workers are created.
  void AddWorkerInitializer(std::function<void(Worker &)> initializer);

  // Initialization APIs. Calls to these methods is only permitted between
  // construction and Initialize().
  // ------------------------------------------------------------------------ //
  void InitializeNodes(int node_count);
  void InitializeHalDriver(std::string_view moniker,
                           iree::hal_driver_ptr driver);
  void InitializeHalDevice(std::unique_ptr<Device> device);
  void FinishInitialization();

 private:
  void AssertNotInitialized() SHORTFIN_REQUIRES_LOCK(lock_) {
    if (initialized_) {
      throw std::logic_error(
          "System::Initialize* methods can only be called during "
          "initialization");
    }
  }
  void AssertRunning() SHORTFIN_REQUIRES_LOCK(lock_) {
    if (!initialized_ || shutdown_) {
      throw std::logic_error(
          "System manipulation methods can only be called when initialized and "
          "not shutdown");
    }
  }

  // Allocates a process in the process table and returns its new pid.
  // This is done on process construction. Note that it acquires the
  // system lock and is non-reentrant.
  int64_t AllocateProcess(detail::BaseProcess *);
  // Deallocates a process by pid. This is done on process destruction. Note
  // that is acquires the system lock and is non-reentrant.
  void DeallocateProcess(int64_t pid);

  // Calls each registered worker initializer.
  void InitializeWorker(Worker &worker);

  const iree_allocator_t host_allocator_;

  string_interner interner_;
  iree::slim_mutex lock_;

  // NUMA nodes relevant to this system.
  std::vector<Node> nodes_;

  // Map of retained hal drivers. These will be released as one of the
  // last steps of destruction. There are some ancillary uses for drivers
  // after initialization, but mainly this is for keeping them alive.
  std::unordered_map<std::string_view, iree::hal_driver_ptr> hal_drivers_;

  // Map of device name to a SystemDevice. Note that devices are immortal and
  // enumerated at initialization time. As such, they are accessed without
  // locking.
  std::vector<std::unique_ptr<Device>> retained_devices_;
  std::unordered_map<std::string_view, Device *> named_devices_;
  std::vector<Device *> devices_;

  // VM management.
  iree::vm_instance_ptr vm_instance_;

  // Global blocking executor.
  BlockingExecutor blocking_executor_;

  // Queues.
  std::vector<std::shared_ptr<Queue>> queues_ SHORTFIN_GUARDED_BY(lock_);
  std::unordered_map<std::string_view, Queue *> queues_by_name_
      SHORTFIN_GUARDED_BY(lock_);

  // Workers.
  std::vector<std::unique_ptr<Worker>> workers_ SHORTFIN_GUARDED_BY(lock_);
  std::vector<std::function<void(Worker &)>> worker_initializers_;
  std::unordered_map<std::string_view, Worker *> workers_by_name_
      SHORTFIN_GUARDED_BY(lock_);

  // Process management.
  int next_pid_ SHORTFIN_GUARDED_BY(lock_) = 1;
  std::unordered_map<int, detail::BaseProcess *> processes_by_pid_
      SHORTFIN_GUARDED_BY(lock_);

  // Whether initialization is complete. If true, various low level
  // mutations are disallowed.
  bool initialized_ SHORTFIN_GUARDED_BY(lock_) = false;
  bool shutdown_ SHORTFIN_GUARDED_BY(lock_) = false;

  friend class detail::BaseProcess;
};
using SystemPtr = std::shared_ptr<System>;

// Base class for configuration objects for setting up a System.
class SHORTFIN_API SystemBuilder {
 public:
  SystemBuilder(iree_allocator_t host_allocator,
                ConfigOptions config_options = {})
      : host_allocator_(host_allocator),
        config_options_(std::move(config_options)) {}
  SystemBuilder() : SystemBuilder(iree_allocator_system()) {}
  virtual ~SystemBuilder() = default;

  // Creates a SystemBuilder subclass for a given named system (i.e.
  // "hostcpu", "amdgpu", etc).
  static std::unique_ptr<SystemBuilder> ForSystem(
      iree_allocator_t host_allocator, std::string_view system_type,
      ConfigOptions config_options = {});

  iree_allocator_t host_allocator() { return host_allocator_; }
  const ConfigOptions &config_options() const { return config_options_; }

  // Construct a System
  virtual SystemPtr CreateSystem() = 0;

 protected:
  // Uses the iree_hal_configure_allocator_from_specs() API to configure
  // allocators for a device. The specs are parsed from the given config_key
  // if it exists and take the form:
  //   some_allocator
  //   some_allocator:key=value
  //   some_allocator:key=value,key=value
  //   some_allocator:key=value,key=value;other_allocator:key=value
  void ConfigureAllocators(const std::vector<std::string> &specs,
                           iree_hal_device_t *device,
                           std::string_view device_debug_desc);

  // Gets a list of allocator specs from the config. If `specific_config_key`
  // is given, this will be consulted first and used if available. Otherwise,
  // "allocators" will be used. For SystemBuilders that handle multiple
  // device types, the specific key will be something like "amdgpu_allocators"
  // or "hostcpu_allocators" and will be used to allow independently scoped
  // allocator specs.
  std::vector<std::string> GetConfigAllocatorSpecs(
      std::optional<std::string_view> specific_config_key);

 private:
  const iree_allocator_t host_allocator_;
  ConfigOptions config_options_;
};

}  // namespace shortfin::local

#endif  // SHORTFIN_LOCAL_SYSTEM_H
